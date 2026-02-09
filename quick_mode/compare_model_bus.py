import sys
import os
import contextlib
import warnings
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

      
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@contextlib.contextmanager
def suppress_stderr():
    old_stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        try:
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)

sys.path.append('/content/drive/MyDrive/Signal_Final_Project')
sys.path.append('/content/drive/MyDrive/Signal_Final_Project/MedSAM')
warnings.filterwarnings('ignore')

                
import config_quick as config
from models_quick import (
    load_medsam, 
    load_text_encoder, 
    load_fusion_model
)
from data_loaders_quick import create_bus_dataloaders
from utils_quick import precompute_text_embeddings, compute_dice, compute_iou
import torch.nn.functional as F
from skimage import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    
with suppress_stderr():  
    medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)
    tokenizer, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)

                            
birads_embeddings = precompute_text_embeddings(
    config.BIRADS_PROMPTS, tokenizer, text_encoder, device
)

                    
print("   Loading V2 model...")
fusion_model_v2 = load_fusion_model(
    version="v2",
    save_path=f"{config.RESULTS_DIR}/fusion_model_bus_best.pth",
    device=device
)
fusion_model_v2.eval()

print("   Loading V4 model...")
fusion_model_v4 = load_fusion_model(
    version="v4",
    save_path=f"{config.RESULTS_DIR}/fusion_model_bus_v4_FULL_best.pth",
    device=device
)
fusion_model_v4.eval()

                                        
embeddings_save_path = f"{config.RESULTS_DIR}/bus_image_embeddings.pt"
img_embeddings_dict = None
if os.path.exists(embeddings_save_path):
    print(f"   Loading cached embeddings from {embeddings_save_path}")
    img_embeddings_dict = torch.load(embeddings_save_path)

                     
train_loader, val_loader = create_bus_dataloaders(
    csv_path=config.BUS_CSV_PATH,
    base_path=config.BUS_DATA_ROOT,
    text_embeddings_dict=birads_embeddings,
    train_split=0.8,
    batch_size=4,
    precomputed_img_embeddings=img_embeddings_dict
)


@torch.no_grad()
def evaluate_model(medsam_model, fusion_model, data_loader, device, model_name="Model"):

    medsam_model.eval()
    if fusion_model is not None:
        fusion_model.eval()
    
    results_list = []
    
    for batch in tqdm(data_loader, desc=f"Evaluating {model_name}"):
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()
        bboxes = batch['bbox'].to(device)
        text_embeds = batch['text_embed'].to(device)
        
                              
        if 'image_embedding' in batch:
            img_embeddings = batch['image_embedding'].to(device)
        else:
            img_embeddings = medsam_model.image_encoder(images)
        
                                                   
        if fusion_model is not None:
            fusion_output = fusion_model(img_embeddings, text_embeds)
            if isinstance(fusion_output, tuple):
                enhanced_embeddings = fusion_output[0]
            else:
                enhanced_embeddings = fusion_output
        else:
                                  
            enhanced_embeddings = img_embeddings
        
                               
        sparse_emb, dense_emb = medsam_model.prompt_encoder(
            points=None,
            boxes=bboxes[:, None, :],
            masks=None
        )
        
                       
        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=enhanced_embeddings,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        
                          
        low_res_pred = torch.sigmoid(low_res_logits)
        preds_1024 = F.interpolate(
            low_res_pred,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        
                              
        for i in range(len(masks)):
            pred_1024 = preds_1024[i, 0].cpu().numpy() > 0.5
            gt_mask = masks[i]
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze()
            
                                     
            H, W = batch['original_size'][0][i].item(), batch['original_size'][1][i].item()
            pred_resized = transform.resize(
                pred_1024.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            
            gt_mask_resized = transform.resize(
                gt_mask.astype(float),
                (H, W),
                order=0,
                preserve_range=True,
                anti_aliasing=False
            ) > 0.5
            
                             
            dice = compute_dice(pred_resized, gt_mask_resized)
            iou = compute_iou(pred_resized, gt_mask_resized)
            
            result = {
                'dice': dice,
                'iou': iou,
                'image_id': batch['image_id'][i],
                'birads': batch['birads'][i].item() if torch.is_tensor(batch['birads'][i]) else batch['birads'][i]
            }
            results_list.append(result)
    
    mean_dice = np.mean([r['dice'] for r in results_list])
    mean_iou = np.mean([r['iou'] for r in results_list])
    
    return results_list, mean_dice, mean_iou

                           
print("\n   Evaluating Baseline (MedSAM without text)...")
baseline_results, baseline_dice, baseline_iou = evaluate_model(
    medsam_model, None, val_loader, device, "Baseline"
)

print("\n   Evaluating V2 (Text-Guided with Gate)...")
v2_results, v2_dice, v2_iou = evaluate_model(
    medsam_model, fusion_model_v2, val_loader, device, "V2"
)

print("\n   Evaluating V4 (Hierarchical Fusion)...")
v4_results, v4_dice, v4_iou = evaluate_model(
    medsam_model, fusion_model_v4, val_loader, device, "V4"
)

                       

print(f"Baseline MedSAM:")
print(f"  Mean Dice: {baseline_dice:.4f} ± {np.std([r['dice'] for r in baseline_results]):.4f}")
print(f"  Mean IoU:  {baseline_iou:.4f} ± {np.std([r['iou'] for r in baseline_results]):.4f}")

print(f"\nV2 Text-Guided:")
print(f"  Mean Dice: {v2_dice:.4f} ± {np.std([r['dice'] for r in v2_results]):.4f}")
print(f"  Mean IoU:  {v2_iou:.4f} ± {np.std([r['iou'] for r in v2_results]):.4f}")
print(f"  Improvement over Baseline: {(v2_dice - baseline_dice):.4f} ({(v2_dice - baseline_dice)/baseline_dice*100:.2f}%)")

print(f"\nV4 Hierarchical:")
print(f"  Mean Dice: {v4_dice:.4f} ± {np.std([r['dice'] for r in v4_results]):.4f}")
print(f"  Mean IoU:  {v4_iou:.4f} ± {np.std([r['iou'] for r in v4_results]):.4f}")
print(f"  Improvement over Baseline: {(v4_dice - baseline_dice):.4f} ({(v4_dice - baseline_dice)/baseline_dice*100:.2f}%)")
print(f"  Improvement over V2: {(v4_dice - v2_dice):.4f} ({(v4_dice - v2_dice)/v2_dice*100:.2f}%)")

                      
results_df = pd.DataFrame({
    'Model': ['Baseline', 'V2', 'V4'],
    'Mean_Dice': [baseline_dice, v2_dice, v4_dice],
    'Std_Dice': [
        np.std([r['dice'] for r in baseline_results]),
        np.std([r['dice'] for r in v2_results]),
        np.std([r['dice'] for r in v4_results])
    ],
    'Mean_IoU': [baseline_iou, v2_iou, v4_iou],
    'Std_IoU': [
        np.std([r['iou'] for r in baseline_results]),
        np.std([r['iou'] for r in v2_results]),
        np.std([r['iou'] for r in v4_results])
    ]
})
results_df.to_csv(f"{config.RESULTS_DIR}/model_comparison_overall.csv", index=False)


                                                       
baseline_df = pd.DataFrame(baseline_results)
v2_df = pd.DataFrame(v2_results)
v4_df = pd.DataFrame(v4_results)

                               
merged_df = baseline_df[['image_id', 'dice']].rename(columns={'dice': 'baseline_dice'})
merged_df = merged_df.merge(
    v2_df[['image_id', 'dice']].rename(columns={'dice': 'v2_dice'}),
    on='image_id'
)
merged_df = merged_df.merge(
    v4_df[['image_id', 'dice', 'birads']].rename(columns={'dice': 'v4_dice'}),
    on='image_id'
)

                            
                                                    
merged_df['improvement'] = merged_df['v4_dice'] - merged_df['baseline_dice']
best_improvement = merged_df.nlargest(3, 'improvement')

                                    
worst_cases = merged_df.nsmallest(3, 'v4_dice')

                                   
merged_df['v4_better_than_v2'] = merged_df['v4_dice'] - merged_df['v2_dice']
progressive_improvement = merged_df[
    (merged_df['v4_dice'] > merged_df['v2_dice']) & 
    (merged_df['v2_dice'] > merged_df['baseline_dice'])
].nlargest(3, 'v4_better_than_v2')

                                                     
sample_dict = {}
for birads in [2, 3, 4, 5]:
    birads_samples = merged_df[merged_df['birads'] == birads]
    if len(birads_samples) > 0:
        sample_dict[f'birads_{birads}'] = birads_samples.sample(n=min(2, len(birads_samples)))

                              
selected_samples = pd.concat([
    best_improvement,
    worst_cases,
    progressive_improvement,
    *sample_dict.values()
], ignore_index=True).drop_duplicates(subset=['image_id'])

@torch.no_grad()
def get_prediction(medsam_model, fusion_model, image, bbox, text_embed, device):
                 
    img_embedding = medsam_model.image_encoder(image.unsqueeze(0).to(device))
                 
    if fusion_model is not None:
        fusion_output = fusion_model(img_embedding, text_embed.unsqueeze(0).to(device))
        if isinstance(fusion_output, tuple):
            enhanced_embedding = fusion_output[0]
        else:
            enhanced_embedding = fusion_output
    else:
        enhanced_embedding = img_embedding
    
                           
    bbox_tensor = bbox.unsqueeze(0).to(device)
    sparse_emb, dense_emb = medsam_model.prompt_encoder(
        points=None,
        boxes=bbox_tensor[:, None, :],
        masks=None
    )
    
             
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=enhanced_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    
              
    pred = torch.sigmoid(low_res_logits)
    pred_1024 = F.interpolate(
        pred,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    
    return pred_1024[0, 0].cpu().numpy() > 0.5

def visualize_comparison(image, gt_mask, pred_baseline, pred_v2, pred_v4, 
                         dice_baseline, dice_v2, dice_v4, 
                         image_id, birads, save_path):  
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
                                         
    if image.ndim == 3 and image.shape[0] == 3:
        image_display = image.transpose(1, 2, 0)
    else:
        image_display = image
    
                                 
    if image_display.max() > 1:
        image_display = image_display / 255.0
    
                                          
    axes[0, 0].imshow(image_display, cmap='gray' if image_display.ndim == 2 else None)
    axes[0, 0].imshow(pred_baseline, alpha=0.5, cmap='Reds')
    axes[0, 0].set_title(f'Baseline\nDice: {dice_baseline:.4f}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image_display, cmap='gray' if image_display.ndim == 2 else None)
    axes[0, 1].imshow(pred_v2, alpha=0.5, cmap='Blues')
    axes[0, 1].set_title(f'V2 (Text-Guided)\nDice: {dice_v2:.4f}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(image_display, cmap='gray' if image_display.ndim == 2 else None)
    axes[0, 2].imshow(pred_v4, alpha=0.5, cmap='Greens')
    axes[0, 2].set_title(f'V4 (Hierarchical)\nDice: {dice_v4:.4f}', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
                       
    axes[1, 0].imshow(pred_baseline, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].contour(gt_mask, colors='yellow', linewidths=2)
    axes[1, 0].set_title('Baseline Mask\n(Yellow = GT)', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_v2, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].contour(gt_mask, colors='yellow', linewidths=2)
    axes[1, 1].set_title('V2 Mask\n(Yellow = GT)', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_v4, cmap='Greens', vmin=0, vmax=1)
    axes[1, 2].contour(gt_mask, colors='yellow', linewidths=2)
    axes[1, 2].set_title('V4 Mask\n(Yellow = GT)', fontsize=10)
    axes[1, 2].axis('off')
    
                   
    improvement_v2 = dice_v2 - dice_baseline
    improvement_v4 = dice_v4 - dice_baseline
    fig.suptitle(
        f'Image ID: {image_id} | BI-RADS: {birads}\n'
        f'V2 Improvement: {improvement_v2:+.4f} | V4 Improvement: {improvement_v4:+.4f}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

                         
vis_dir = f"{config.RESULTS_DIR}/visualizations"
os.makedirs(vis_dir, exist_ok=True)


image_id_to_data = {}

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Building lookup", leave=False):
        for i in range(len(batch['image'])):
            img_id = batch['image_id'][i]
            birads_val = batch['birads'][i]
            if torch.is_tensor(birads_val):
                birads_val = birads_val.item()
            
            image_id_to_data[img_id] = {
                'image': batch['image'][i],
                'mask': batch['mask'][i],
                'bbox': batch['bbox'][i],
                'text_embed': batch['text_embed'][i],
                'birads': birads_val,
                'original_size': (batch['original_size'][0][i].item(), batch['original_size'][1][i].item())
            }

                   
sample_counter = 0
for idx, row in tqdm(selected_samples.iterrows(), total=len(selected_samples), desc="Creating visualizations"):
    image_id = row['image_id']
    
                                      
    if image_id not in image_id_to_data:
        print(f"   Warning: image_id {image_id} not found, skipping...")
        continue
    
    sample_data = image_id_to_data[image_id]
    image = sample_data['image']
    mask = sample_data['mask']
    bbox = sample_data['bbox']
    text_embed = sample_data['text_embed']
    birads = sample_data['birads']
    H, W = sample_data['original_size']
    
                       
    if mask.ndim == 3:
        mask = mask.squeeze()
    
                                           
    pred_baseline = get_prediction(medsam_model, None, image, bbox, text_embed, device)
    pred_v2 = get_prediction(medsam_model, fusion_model_v2, image, bbox, text_embed, device)
    pred_v4 = get_prediction(medsam_model, fusion_model_v4, image, bbox, text_embed, device)
    
                                         
    pred_baseline_resized = transform.resize(pred_baseline, (H, W), order=0, preserve_range=True, anti_aliasing=False) > 0.5
    pred_v2_resized = transform.resize(pred_v2, (H, W), order=0, preserve_range=True, anti_aliasing=False) > 0.5
    pred_v4_resized = transform.resize(pred_v4, (H, W), order=0, preserve_range=True, anti_aliasing=False) > 0.5
    mask_resized = transform.resize(mask.cpu().numpy() if torch.is_tensor(mask) else mask, (H, W), order=0, preserve_range=True, anti_aliasing=False) > 0.5
    
                         
    dice_baseline = compute_dice(pred_baseline_resized, mask_resized)
    dice_v2 = compute_dice(pred_v2_resized, mask_resized)
    dice_v4 = compute_dice(pred_v4_resized, mask_resized)
    
                                 
    image_np = image.cpu().numpy() if torch.is_tensor(image) else image
    
               
    save_path = f"{vis_dir}/comparison_{sample_counter:03d}_{image_id}.png"
    visualize_comparison(
        image_np, mask_resized,
        pred_baseline_resized, pred_v2_resized, pred_v4_resized,
        dice_baseline, dice_v2, dice_v4,
        image_id, birads,
        save_path
    )
    
    sample_counter += 1
       
                                                                                  
         
                 
         
                           
                           
                                                                                   
    
                    
                                  
                                                                                         
                                              
                                                   
               
                                                
           
                                            
    
                             
                                                
                                                          
                      
                                        
                    
       
    
               
                                                    
                                              
                                                              
                                              
                                            
                                 
       
    
                
                                          
                                
               
                            
                          
                              
       
    
                                                

                                                                            
                                                            
                                                        
         
                  
         
                                                      
    
                                           
                                                 
                                                  
           
                               
    
                                   
                                 
                                               
    
                                            
                                                                                        
                                                              
                                                                                                  
                            
    
                                                                                        
                                                         
                                                                                                    
                            
    
                                                                                        
                                                          
                                                                                                     
                            
    
                         
                                                                   
                                                                
                                                                       
                            
    
                                                              
                                                                
                                                                 
                            
    
                                                               
                                                                
                                                                 
                            
    
                     
                                              
                                              
                   
                                                       
                                                                                           
                      
                           
       
    
                        
                                                          
                 

                           
                                                  
                                     

                                                
                                                                               

                                                              
                                  
                    

                                                                                                                 
                                
    
                                       
                       
                                       
                                                                                                        
                                          
                          
                 
    
                            
                  
    
                      
                                      
                             
                           
                           
                                       
                               
    
                         
                        
                               
    
                                             
                                                                                         
                                                                                              
                                                                                              
    
                         
                                    
    
                                           
                                                                                                                              
                                                                                                                  
                                                                                                                  
                                                                                                                    
    
                           
                                                                       
                                                           
                                                           
    
                                   
                              
    
                 
                                                                             
                           
                                 
                                                                  
                                          
                           
                   
       
    
                         

                                                       
                                  

                                                              
                       
                                                              
print("\n[Bonus] Generating Summary Plots...")

                        
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

dice_data = [
    [r['dice'] for r in baseline_results],
    [r['dice'] for r in v2_results],
    [r['dice'] for r in v4_results]
]

axes[0].boxplot(dice_data, labels=['Baseline', 'V2', 'V4'])
axes[0].set_ylabel('Dice Score', fontsize=12)
axes[0].set_title('Dice Score Distribution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

                             
models = ['Baseline', 'V2', 'V4']
means = [baseline_dice, v2_dice, v4_dice]
stds = [
    np.std([r['dice'] for r in baseline_results]),
    np.std([r['dice'] for r in v2_results]),
    np.std([r['dice'] for r in v4_results])
]

x_pos = np.arange(len(models))
axes[1].bar(x_pos, means, yerr=stds, capsize=10, color=['red', 'blue', 'green'], alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(models)
axes[1].set_ylabel('Mean Dice Score', fontsize=12)
axes[1].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

                          
for i, (mean, std) in enumerate(zip(means, stds)):
    axes[1].text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{config.RESULTS_DIR}/model_comparison_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print("✓ Summary plots saved")

                                                              
               
                                                              
print("\n" + "="*70)
print("EXPERIMENT COMPLETED!")
print("="*70)
print(f"Results saved to: {config.RESULTS_DIR}")
print(f"  - Overall metrics: model_comparison_overall.csv")
print(f"  - Visualizations: {vis_dir}/")
print(f"  - Summary plot: model_comparison_summary.png")
print("="*70)