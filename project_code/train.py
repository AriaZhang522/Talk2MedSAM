
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils import compute_dice

def normalize_mask_for_loss(masks):

    masks = masks.float()

    if masks.ndim == 2:
        # [H, W] -> [1, 1, H, W]
        masks = masks.unsqueeze(0).unsqueeze(0)

    elif masks.ndim == 3:
        # [B, H, W] -> [B, 1, H, W]
        masks = masks.unsqueeze(1)

    elif masks.ndim == 4:

        pass

    elif masks.ndim == 5:


        B = masks.shape[0]
        H, W = masks.shape[-2], masks.shape[-1]
        C = int(masks.numel() / (B * H * W))

        print(f"[normalize_mask_for_loss] Warning: got 5D mask {masks.shape}, "
              f"reshaping to (B={B}, C={C}, H={H}, W={W})")

        masks = masks.view(B, C, H, W)

        if C > 1:
            masks = masks[:, :1]

    else:
        raise ValueError(f"Unexpected mask ndim={masks.ndim}, shape={masks.shape}")

    return masks

class CombinedLoss(nn.Module):

    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        """
        Soft Dice Loss
        pred: [B, 1, H, W] logits
        target: [B, 1, H, W] binary
        """
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        return 1 - dice.mean()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


# ============================================================
# Training Function
# ============================================================
def train_fusion_model(medsam_model, fusion_model, train_loader, val_loader, 
                      num_epochs, learning_rate, device, save_path, use_precomputed=True):

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=learning_rate)
    criterion = CombinedLoss()
    
    best_val_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }
    
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        # ==================== Training ====================
        fusion_model.train()
        train_losses = []
        train_dices = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            bboxes = batch['bbox'].to(device)
            text_embeds = batch['text_embed'].to(device)
            
            if use_precomputed and 'image_embedding' in batch:
                img_embeddings = batch['image_embedding'].to(device)
            else:
                with torch.no_grad():
                    img_embeddings = medsam_model.image_encoder(images)
            

            enhanced_embeddings = fusion_model(img_embeddings, text_embeds)
            
            # Get prompt embeddings
            with torch.no_grad():
                sparse_emb, dense_emb = medsam_model.prompt_encoder(
                    points=None,
                    boxes=bboxes[:, None, :],
                    masks=None
                )
            
            # Predict masks
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=enhanced_embeddings,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            
            # Resize target masks to match prediction size (256x256)
            target_size = low_res_logits.shape[-2:]
            # masks = masks.float()
            # if masks.ndim == 3:
            #      masks = masks.unsqueeze(1)
            # elif masks.ndim == 4:
            #     pass
            # elif masks.ndim == 5:
            #     masks = masks.squeeze(2)
            masks = normalize_mask_for_loss(masks)

            masks_resized = F.interpolate(
                masks,
                size=target_size,
                mode='nearest'
            )
            
            # Compute loss
            loss = criterion(low_res_logits, masks_resized)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            train_losses.append(loss.item())
            
            # Compute Dice on original resolution
            # with torch.no_grad():
            #     preds = torch.sigmoid(low_res_logits) > 0.5
            #     preds_1024 = F.interpolate(
            #         preds.float(),
            #         size=(1024, 1024),
            #         mode='bilinear',
            #         align_corners=False
            #     )
                
            #     for i in range(len(masks)):
            #         pred_np = preds_1024[i, 0].cpu().numpy()
            #         mask_np = masks[i].cpu().numpy()
                    
            #         # Resize to original size
            #         H, W = batch['original_size'][0][i].item(), batch['original_size'][1][i].item()
            #         from skimage import transform
            #         pred_resized = transform.resize(pred_np, (H, W), order=0) > 0.5
                    
            #         dice = compute_dice(pred_resized, mask_np)
            #         train_dices.append(dice)
            with torch.no_grad():
                  preds = torch.sigmoid(low_res_logits) > 0.5      # [B,1,h,w]

                  mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                  preds_up = F.interpolate(
                      preds.float(),
                      size=(mask_h, mask_w),
                      mode='nearest'
                  )
                  for i in range(masks.shape[0]):

                      pred_np = preds_up[i, 0].cpu().numpy() > 0.5

                      mask_i = masks[i]
                      if mask_i.ndim == 3:
                          mask_i = mask_i[0]
                      mask_np = (mask_i.cpu().numpy() > 0.5)

                      dice = compute_dice(pred_np, mask_np)
                      train_dices.append(dice)
                      
                  pbar.set_postfix({
                'loss': f"{np.mean(train_losses):.4f}",
                'dice': f"{np.mean(train_dices):.4f}"
            })
        
        # ==================== Validation ====================
        fusion_model.eval()
        val_losses = []
        val_dices = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                bboxes = batch['bbox'].to(device)
                text_embeds = batch['text_embed'].to(device)
                
                # Get image embeddings
                # img_embeddings = medsam_model.image_encoder(images)
                if use_precomputed and 'image_embedding' in batch:
                    img_embeddings = batch['image_embedding'].to(device)
                else:
                    # Get image embeddings
                    img_embeddings = medsam_model.image_encoder(images)
                            
                # Apply fusion
                enhanced_embeddings = fusion_model(img_embeddings, text_embeds)
                
                # Get prompt embeddings
                sparse_emb, dense_emb = medsam_model.prompt_encoder(
                    points=None,
                    boxes=bboxes[:, None, :],
                    masks=None
                )
                
                # Predict
                low_res_logits, _ = medsam_model.mask_decoder(
                    image_embeddings=enhanced_embeddings,
                    image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                # Resize target
                target_size = low_res_logits.shape[-2:]
                masks = normalize_mask_for_loss(masks) 
                # masks = masks.float()
                # if masks.ndim == 3:           # [B, H, W]
                #     masks = masks.unsqueeze(1)
                # elif masks.ndim == 4:         # [B, 1, H, W]
                #     pass
                # else:
                #     raise ValueError(f"Unexpected mask shape in val loop: {masks.shape}")

                masks_resized = F.interpolate(
                    masks,
                    size=target_size,
                    mode='nearest'
                )
                
                # Compute loss
                loss = criterion(low_res_logits, masks_resized)
                val_losses.append(loss.item())
                
                # Compute Dice
                preds = torch.sigmoid(low_res_logits) > 0.5

                mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                preds_up = F.interpolate(
                    preds.float(),
                    size=(mask_h, mask_w),
                    mode='nearest'
                )

                for i in range(masks.shape[0]):
                    pred_np = preds_up[i, 0].cpu().numpy() > 0.5

                    mask_i = masks[i]
                    if mask_i.ndim == 3:
                        mask_i = mask_i[0]
                    mask_np = (mask_i.cpu().numpy() > 0.5)

                    dice = compute_dice(pred_np, mask_np)
                    val_dices.append(dice)
                
                pbar.set_postfix({
                    'loss': f"{np.mean(val_losses):.4f}",
                    'dice': f"{np.mean(val_dices):.4f}"
                })
        
        # ==================== Epoch Summary ====================
        epoch_train_loss = np.mean(train_losses)
        epoch_train_dice = np.mean(train_dices)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_dice = np.mean(val_dices)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_dice'].append(epoch_train_dice)
        history['val_loss'].append(epoch_val_loss)
        history['val_dice'].append(epoch_val_dice)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Dice: {epoch_train_dice:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f}, Val Dice:   {epoch_val_dice:.4f}")
        
        # Save best model
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'gate_value': fusion_model.gate.item() if hasattr(fusion_model, 'gate') else None
            }, save_path)
            print(f"  ✓ New best model saved! (Val Dice: {best_val_dice:.4f})")
        
        print("=" * 70)
    
    print(f"\nTraining completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    
    return history