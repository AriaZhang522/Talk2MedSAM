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
from models_quick import (
    load_medsam, 
    load_text_encoder, 
    create_fusion_model,
    load_fusion_model
)
from data_loaders_quick import create_nsclc_dataloaders
from utils_quick import precompute_text_embeddings, compute_dice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER_MODEL)
_, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)

                                    
t_stage_embeddings = precompute_text_embeddings(
    config.T_STAGE_PROMPTS, tokenizer, text_encoder, device
)
histology_embeddings = precompute_text_embeddings(
    config.HISTOLOGY_PROMPTS, tokenizer, text_encoder, device
)
text_embeddings_dict = {
    't_stage': t_stage_embeddings,
    'histology': histology_embeddings
}

medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)


          
v2_model_path = f"{config.RESULTS_DIR}/fusion_model_2_nsclc_best.pth"
                                                                                                           
fusion_model_v2 = load_fusion_model(
    version="v2",
    save_path=v2_model_path,
    device=device
)
fusion_model_v2.eval()

          
v4_model_path = f"{config.RESULTS_DIR}/fusion_model_nsclc_v4_FULL_best.pth"
fusion_model_v4 = create_fusion_model(version="v4").to(device)
checkpoint = torch.load(v4_model_path, map_location=device, weights_only=False)
fusion_model_v4.load_state_dict(checkpoint['model_state_dict'])
fusion_model_v4.eval()

_, val_loader = create_nsclc_dataloaders(
    csv_path=config.NSCLC_CSV_PATH,
    data_root=config.NSCLC_DATA_ROOT,
    text_embeddings_dict=text_embeddings_dict,
    train_split=0.8,
    batch_size=1
)
                                                    
@torch.no_grad()
def get_prediction(medsam_model, fusion_model, img_emb, text_embed, bbox, use_fusion=True):
\
\
       
    if use_fusion:
        fusion_output = fusion_model(img_emb, text_embed)
        if isinstance(fusion_output, tuple):
            enhanced_emb = fusion_output[0]
        else:
            enhanced_emb = fusion_output
    else:
                              
        enhanced_emb = img_emb
    
    sparse, dense = medsam_model.prompt_encoder(
        points=None, 
        boxes=bbox[:, None, :], 
        masks=None
    )
    
    logits, _ = medsam_model.mask_decoder(
        image_embeddings=enhanced_emb,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False
    )
    
                      
    from torch.nn.functional import interpolate
    logits = interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
    mask = (torch.sigmoid(logits) > 0.5).cpu().numpy().squeeze()
    
    return mask


def visualize_nsclc_comparison(medsam_model, fusion_v2, fusion_v4, val_loader, device, 
                               t_stage_emb, histology_emb, num_samples=3):

       
    
    medsam_model.eval()
    fusion_v2.eval()
    fusion_v4.eval()
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= num_samples:
            break
        
              
        img = batch['image'].to(device)
        bbox = batch['bbox'].to(device)
        gt_mask_original = batch['mask'][0].numpy()
        if gt_mask_original.ndim == 3:
            gt_mask_original = gt_mask_original[0]
        
                           
        gt_mask_1024 = transform.resize(
            gt_mask_original, (1024, 1024), 
            order=0, preserve_range=True, anti_aliasing=False
        ) > 0.5
        
                
        patient_id = batch['patient_id'][0]
        t_stage = batch['t_stage'][0]
        histology = batch['histology'][0]
        
        print(f"\n   Sample {sample_count + 1}: {patient_id} | T-stage: {t_stage} | Histology: {histology}")
        
                     
        t_embed = t_stage_emb[t_stage].unsqueeze(0).to(device)
        h_embed = histology_emb[histology].unsqueeze(0).to(device)
        text_embed = (t_embed + h_embed) / 2      
        
                
        img_emb = medsam_model.image_encoder(img)
        
                        
                           
        mask_baseline = get_prediction(medsam_model, None, img_emb, None, bbox, use_fusion=False)
        dice_baseline = compute_dice(mask_baseline, gt_mask_1024)
        
                      
        mask_v2 = get_prediction(medsam_model, fusion_v2, img_emb, text_embed, bbox, use_fusion=True)
        dice_v2 = compute_dice(mask_v2, gt_mask_1024)
        
                      
        mask_v4 = get_prediction(medsam_model, fusion_v4, img_emb, text_embed, bbox, use_fusion=True)
        dice_v4 = compute_dice(mask_v4, gt_mask_1024)
        
        print(f"      Dice → Baseline: {dice_baseline:.4f} | V2: {dice_v2:.4f} | V4: {dice_v4:.4f}")
        
                     
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
              
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
                                                          
                               
        axes[0, 0].imshow(img_np, cmap='gray')
        masked_gt = np.ma.masked_where(gt_mask_1024 == 0, gt_mask_1024)
        axes[0, 0].imshow(masked_gt, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[0, 0].contour(gt_mask_1024, colors='red', linewidths=1.5)
        axes[0, 0].set_title(f'Ground Truth\n(Red)', fontsize=12, fontweight='bold', color='darkred')
        axes[0, 0].axis('off')
        
                           
        axes[0, 1].imshow(img_np, cmap='gray')
        masked_baseline = np.ma.masked_where(mask_baseline == 0, mask_baseline)
        axes[0, 1].imshow(masked_baseline, cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
        axes[0, 1].contour(mask_baseline, colors='orange', linewidths=1.5)
        axes[0, 1].set_title(f'Baseline\nDice: {dice_baseline:.4f}', fontsize=12, fontweight='bold', color='darkorange')
        axes[0, 1].axis('off')
        
                     
        axes[0, 2].imshow(img_np, cmap='gray')
        masked_v2 = np.ma.masked_where(mask_v2 == 0, mask_v2)
        axes[0, 2].imshow(masked_v2, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        axes[0, 2].contour(mask_v2, colors='blue', linewidths=1.5)
        axes[0, 2].set_title(f'V2 (Text-Guided)\nDice: {dice_v2:.4f}', fontsize=12, fontweight='bold', color='darkblue')
        axes[0, 2].axis('off')
        
                     
        axes[0, 3].imshow(img_np, cmap='gray')
        masked_v4 = np.ma.masked_where(mask_v4 == 0, mask_v4)
        axes[0, 3].imshow(masked_v4, cmap='Greens', alpha=0.6, vmin=0, vmax=1)
        axes[0, 3].contour(mask_v4, colors='lime', linewidths=1.5)
        axes[0, 3].set_title(f'V4 (Hierarchical)\nDice: {dice_v4:.4f}', fontsize=12, fontweight='bold', color='darkgreen')
        axes[0, 3].axis('off')
        
                                                       
                     
        axes[1, 0].imshow(gt_mask_1024, cmap='Reds', vmin=0, vmax=1)
        axes[1, 0].set_title('GT Mask', fontsize=10)
        axes[1, 0].axis('off')
        
                                        
        axes[1, 1].imshow(mask_baseline, cmap='YlOrBr', vmin=0, vmax=1)
        axes[1, 1].contour(gt_mask_1024, colors='red', linewidths=2, linestyles='--')
        axes[1, 1].set_title('Baseline\n(Red = GT)', fontsize=10)
        axes[1, 1].axis('off')
        
                                  
        axes[1, 2].imshow(mask_v2, cmap='Blues', vmin=0, vmax=1)
        axes[1, 2].contour(gt_mask_1024, colors='red', linewidths=2, linestyles='--')
        axes[1, 2].set_title('V2\n(Red = GT)', fontsize=10)
        axes[1, 2].axis('off')
        
                                  
        axes[1, 3].imshow(mask_v4, cmap='Greens', vmin=0, vmax=1)
        axes[1, 3].contour(gt_mask_1024, colors='red', linewidths=2, linestyles='--')
        axes[1, 3].set_title('V4\n(Red = GT)', fontsize=10)
        axes[1, 3].axis('off')
        
             
        improvement_v2 = dice_v2 - dice_baseline
        improvement_v4 = dice_v4 - dice_baseline
        fig.suptitle(
            f'Patient: {patient_id} | T-stage: {t_stage} | Histology: {histology}\n'
            f'V2 Improvement: {improvement_v2:+.4f} | V4 Improvement: {improvement_v4:+.4f}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
              
        save_path = f"nsclc_visualization_sample_{sample_count + 1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
                       
        try:
            from IPython.display import Image, display
            display(Image(filename=save_path))
        except:
            pass
        
        sample_count += 1
                                                           
if __name__ == "__main__":
    visualize_nsclc_comparison(
        medsam_model=medsam_model,
        fusion_v2=fusion_model_v2,
        fusion_v4=fusion_model_v4,
        val_loader=val_loader,
        device=device,
        t_stage_emb=t_stage_embeddings,
        histology_emb=histology_embeddings,
        num_samples=5             
    )