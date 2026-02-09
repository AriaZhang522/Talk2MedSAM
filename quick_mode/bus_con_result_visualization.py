import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings
from skimage import transform

      
sys.path.append('/content/drive/MyDrive/Signal_Final_Project')
sys.path.append('/content/drive/MyDrive/Signal_Final_Project/MedSAM')
warnings.filterwarnings('ignore')

import config_quick as config
from models_quick import load_medsam, load_text_encoder, create_fusion_model
from data_loaders_quick import create_bus_dataloaders
from utils_quick import precompute_text_embeddings, compute_dice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER_MODEL)
_, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)

birads_embeddings = precompute_text_embeddings(
    config.BIRADS_PROMPTS, tokenizer, text_encoder, device
)

medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)
fusion_model = create_fusion_model(version="v4").to(device)

model_path = f"{config.RESULTS_DIR}/fusion_model_bus_v4_FULL_best.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
fusion_model.load_state_dict(checkpoint['model_state_dict'])
fusion_model.eval()
medsam_model.eval()


_, val_loader = create_bus_dataloaders(
    csv_path=config.BUS_CSV_PATH,
    base_path=config.BUS_DATA_ROOT,
    text_embeddings_dict=birads_embeddings,
    train_split=0.8,
    batch_size=1,
    precomputed_img_embeddings=None
)
                                                   
@torch.no_grad()
def get_prediction_with_text(medsam_model, fusion_model, img, bbox, text_embed, device):
                    
                                    
    if img.ndim == 3:
        img = img.unsqueeze(0)
    
    img_emb = medsam_model.image_encoder(img.to(device))
    
    fused = fusion_model(img_emb, text_embed.unsqueeze(0).to(device))
    if isinstance(fused, tuple):
        fused = fused[0]
    
                    
    if bbox.ndim == 1:
        bbox = bbox.unsqueeze(0)
    
    sparse, dense = medsam_model.prompt_encoder(
        points=None, boxes=bbox[:, None, :].to(device), masks=None
    )
    
    logits, _ = medsam_model.mask_decoder(
        image_embeddings=fused,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False
    )
    
    from torch.nn.functional import interpolate
    logits = interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
    mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze()
    
    return mask


def visualize_multiple_prompts(sample, birads_embeddings, true_birads, medsam_model, fusion_model, device):
 
    img = sample['image']
    bbox = sample['bbox']
    gt_mask = sample['mask'].numpy()
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]
    
    gt_mask_1024 = transform.resize(gt_mask, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False) > 0.5
    
          
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
                        
    birads_options = [2, 3, 4, 5]
    predictions = {}
    dice_scores = {}
    
    for birads in birads_options:
        text_embed = birads_embeddings[birads]
        mask = get_prediction_with_text(medsam_model, fusion_model, img, bbox, text_embed, device)
        predictions[birads] = mask
        dice_scores[birads] = compute_dice(mask, gt_mask_1024)
    
                                      
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
                                 
        
    axes[0, 0].imshow(img_np)
    masked_gt = np.ma.masked_where(gt_mask_1024 == 0, gt_mask_1024)
    axes[0, 0].imshow(masked_gt, cmap='Reds', alpha=0.5)
    axes[0, 0].contour(gt_mask_1024, colors='red', linewidths=2)
    axes[0, 0].set_title(f'Ground Truth\nTrue BI-RADS: {true_birads}', fontsize=12, fontweight='bold', color='darkred')
    axes[0, 0].axis('off')
    
               
    axes[0, 1].imshow(img_np)
    masked = np.ma.masked_where(predictions[2] == 0, predictions[2])
    axes[0, 1].imshow(masked, cmap='Oranges', alpha=0.5)
    axes[0, 1].contour(predictions[2], colors='orange', linewidths=2)
    axes[0, 1].set_title(f'BI-RADS 2 (Benign)\nDice: {dice_scores[2]:.4f}', fontsize=12, color='darkorange')
    axes[0, 1].axis('off')
    
               
    axes[0, 2].imshow(img_np)
    masked = np.ma.masked_where(predictions[3] == 0, predictions[3])
    axes[0, 2].imshow(masked, cmap='YlGn', alpha=0.5)
    axes[0, 2].contour(predictions[3], colors='yellowgreen', linewidths=2)
    axes[0, 2].set_title(f'BI-RADS 3 (Probably Benign)\nDice: {dice_scores[3]:.4f}', fontsize=12, color='olive')
    axes[0, 2].axis('off')
    
                             
               
    axes[1, 0].imshow(img_np)
    masked = np.ma.masked_where(predictions[4] == 0, predictions[4])
    axes[1, 0].imshow(masked, cmap='Blues', alpha=0.5)
    axes[1, 0].contour(predictions[4], colors='blue', linewidths=2)
    axes[1, 0].set_title(f'BI-RADS 4 (Suspicious)\nDice: {dice_scores[4]:.4f}', fontsize=12, color='darkblue')
    axes[1, 0].axis('off')
    
               
    axes[1, 1].imshow(img_np)
    masked = np.ma.masked_where(predictions[5] == 0, predictions[5])
    axes[1, 1].imshow(masked, cmap='Purples', alpha=0.5)
    axes[1, 1].contour(predictions[5], colors='purple', linewidths=2)
    axes[1, 1].set_title(f'BI-RADS 5 (Highly Suspicious)\nDice: {dice_scores[5]:.4f}', fontsize=12, color='purple')
    axes[1, 1].axis('off')
    
                                           
    correct_pred = predictions[true_birads]
                                
    wrong_birads = min(dice_scores.keys(), key=lambda x: dice_scores[x] if x != true_birads else 1.0)
    wrong_pred = predictions[wrong_birads]
    
    diff = correct_pred.astype(float) - wrong_pred.astype(float)
    im = axes[1, 2].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    axes[1, 2].set_title(f'Difference Map\nCorrect ({true_birads}) - Wrong ({wrong_birads})', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    fig.suptitle(
        f'Effect of Different BI-RADS Prompts on Segmentation\n'
        f'Ground Truth: BI-RADS {true_birads} | Image ID: {sample["image_id"]}',
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout()
    save_path = f"bus_counterfactual_all_prompts_{sample['image_id']}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
                                        
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
        
    axes[0].imshow(img_np)
    masked_gt = np.ma.masked_where(gt_mask_1024 == 0, gt_mask_1024)
    axes[0].imshow(masked_gt, cmap='Reds', alpha=0.5)
    axes[0].contour(gt_mask_1024, colors='red', linewidths=2)
    axes[0].set_title(f'Ground Truth\nBI-RADS {true_birads}', fontsize=14, fontweight='bold', color='darkred')
    axes[0].axis('off')
    
             
    axes[1].imshow(img_np)
    masked = np.ma.masked_where(correct_pred == 0, correct_pred)
    axes[1].imshow(masked, cmap='Greens', alpha=0.5)
    axes[1].contour(correct_pred, colors='lime', linewidths=2)
    axes[1].set_title(f'Correct Text (BI-RADS {true_birads})\nDice: {dice_scores[true_birads]:.4f}', fontsize=14, color='darkgreen')
    axes[1].axis('off')
    
           
    axes[2].imshow(img_np)
    masked = np.ma.masked_where(wrong_pred == 0, wrong_pred)
    axes[2].imshow(masked, cmap='Blues', alpha=0.5)
    axes[2].contour(wrong_pred, colors='blue', linewidths=2)
    axes[2].set_title(f'Wrong Text (BI-RADS {wrong_birads})\nDice: {dice_scores[wrong_birads]:.4f}', fontsize=14, color='darkblue')
    axes[2].axis('off')
    
                
    axes[3].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    axes[3].set_title('Difference Map\n(Red=Correct Only, Blue=Wrong Only)', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    fig.suptitle(
        f'Counterfactual Analysis | Image: {sample["image_id"]} | True BI-RADS: {true_birads}',
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout()
    save_path_simple = f"bus_counterfactual_simple_{sample['image_id']}.png"
    plt.savefig(save_path_simple, dpi=150, bbox_inches='tight')
    plt.close()
    
    return dice_scores


num_samples = 3            
sample_count = 0

for batch in val_loader:
    if sample_count >= num_samples:
        break
    
                            
    birads = batch['birads'][0]
    if isinstance(birads, torch.Tensor):
        birads = birads.item()
    
    if birads not in [4, 5]:
        continue
    
    sample = {
        'image': batch['image'][0],
        'mask': batch['mask'][0],
        'bbox': batch['bbox'][0],
        'image_id': batch['image_id'][0],
        'birads': birads
    }
    
    print(f"\n[Sample {sample_count + 1}] Image: {sample['image_id']} | BI-RADS: {birads}")
    
    dice_scores = visualize_multiple_prompts(
        sample, birads_embeddings, birads, medsam_model, fusion_model, device
    )
    
    print(f"   Dice scores across all BI-RADS prompts:")
    for b in [2, 3, 4, 5]:
        print(f"     BI-RADS {b}: {dice_scores[b]:.4f}")
    
    sample_count += 1

print('   - *_all_prompts_*.png:  BI-RADS ')
print('   - *_simple_*.png: （ vs ）')