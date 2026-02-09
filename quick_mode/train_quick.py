import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils_quick import compute_dice
                                                      
def debug_val_one_batch(medsam_model, fusion_model, val_loader, device):
       
    medsam_model.eval()
    fusion_model.eval()
    criterion = CombinedLoss()                

    from utils_quick import compute_dice
    import torch.nn.functional as F
    import numpy as np

    with torch.no_grad():
        batch = next(iter(val_loader))                

        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        bboxes = batch['bbox'].to(device)
        text_embeds = batch['text_embed'].to(device)

                          
        img_embeddings = medsam_model.image_encoder(images)

                   
        enhanced_embeddings = fusion_model(img_embeddings, text_embeds)

                           
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

                                
        target_size = low_res_logits.shape[-2:]

                                                          
        masks = normalize_mask_for_loss(masks)              

        masks_resized = F.interpolate(
            masks,
            size=target_size,
            mode='nearest'
        )

                 
        loss = criterion(low_res_logits, masks_resized).item()

                               
        preds = torch.sigmoid(low_res_logits) > 0.5

        mask_h, mask_w = masks.shape[-2], masks.shape[-1]
        preds_up = F.interpolate(
            preds.float(),
            size=(mask_h, mask_w),
            mode='nearest'
        )

        dices = []
        for i in range(masks.shape[0]):
            pred_np = preds_up[i, 0].cpu().numpy() > 0.5

            mask_i = masks[i]
            if mask_i.ndim == 3:
                mask_i = mask_i[0]
            mask_np = (mask_i.cpu().numpy() > 0.5)

            dices.append(compute_dice(pred_np, mask_np))

        mean_dice = float(np.mean(dices))

        print("\n[DEBUG] One-batch VAL check:")
        print(f"  loss = {loss:.4f}")
        print(f"  dice = {mean_dice:.4f}")
        print(f"  low_res_logits shape: {low_res_logits.shape}")
        print(f"  masks_resized shape:  {masks_resized.shape}")
        return loss, mean_dice

def normalize_mask_for_loss(masks):
 
    masks = masks.float()

    if masks.ndim == 2:
                                
        masks = masks.unsqueeze(0).unsqueeze(0)

    elif masks.ndim == 3:
                                   
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


class ContrastiveLoss(nn.Module):
 
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, img_features, text_embeds, labels):
                               
        img_pooled = F.adaptive_avg_pool2d(img_features, 1)                  
        img_pooled = img_pooled.view(img_pooled.size(0), -1)            
     
        img_pooled = F.normalize(img_pooled, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)
                        
        logits = torch.matmul(img_pooled, text_embeds.T) / self.temperature                                                  
        targets = torch.zeros_like(logits)
        for i in range(len(labels)):
            for j in range(len(labels)):
                                
                label_i = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                label_j = labels[j].item() if torch.is_tensor(labels[j]) else labels[j]
                if label_i == label_j:
                    targets[i, j] = 1.0
        
                             
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return loss



def train_fusion_model(medsam_model, fusion_model, train_loader, val_loader, 
                      num_epochs, learning_rate, device, save_path, use_precomputed=True):
             
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=learning_rate)
    
                     
    seg_criterion = CombinedLoss()               
    con_criterion = ContrastiveLoss()            
    con_weight = 2.0                                
    
    best_val_dice = 0.0
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss': [], 'val_dice': []
    }

    for epoch in range(num_epochs):
                                                            
        fusion_model.train()
        train_losses = []
        train_dices = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            bboxes = batch['bbox'].to(device)
            text_embeds = batch['text_embed'].to(device)
                                       
                                
            if 'birads' in batch:
                labels = batch['birads']           
            elif 't_stage' in batch:
                labels = batch['t_stage']                              
            else:
                labels = None                             
            
                                      
            if use_precomputed and 'image_embedding' in batch:
                img_embeddings = batch['image_embedding'].to(device)
            else:
                with torch.no_grad():
                    img_embeddings = medsam_model.image_encoder(images)
            
                        
            fusion_output = fusion_model(img_embeddings, text_embeds)
            if isinstance(fusion_output, tuple):
                enhanced_embeddings = fusion_output[0]
            else:
                enhanced_embeddings = fusion_output
            
                              
            with torch.no_grad():
                sparse_emb, dense_emb = medsam_model.prompt_encoder(
                    points=None, boxes=bboxes[:, None, :], masks=None
                )
            
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=enhanced_embeddings,
                image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            
                                  
            masks = normalize_mask_for_loss(masks)
            target_size = low_res_logits.shape[-2:]
            masks_resized = F.interpolate(masks, size=target_size, mode='nearest')
            
                                  
                        
            loss_seg = seg_criterion(low_res_logits, masks_resized)
            
                                    
            text_projected = None
            if hasattr(fusion_model, 'text_tokenizer'):                  
                text_hidden = fusion_model.text_tokenizer(text_embeds)                   
                text_tokens = fusion_model.text_token_proj(text_hidden)                  
                                     
                B, _ = text_tokens.shape
                                                         
                C = enhanced_embeddings.shape[1]                
                text_tokens = text_tokens.reshape(B, -1, C)              
                
                                            
                text_projected = text_tokens.mean(dim=1)           
                
            elif hasattr(fusion_model, 'text_coarse'): 
                                      
                text_projected = fusion_model.text_coarse(text_embeds)
                if text_projected.ndim == 3: 
                    text_projected = text_projected.squeeze(1)
                    
            elif hasattr(fusion_model, 'text_proj'):
                          
                text_projected = fusion_model.text_proj(text_embeds)
                if text_projected.ndim > 2:
                    text_projected = text_projected.flatten(1)
            con_loss_val = 0.0
            if text_projected is not None and labels is not None:                
                current_con_weight = 0.5 if hasattr(fusion_model, 'text_tokenizer') else con_weight
                loss_con = con_criterion(enhanced_embeddings, text_projected, labels)             
                loss = loss_seg + current_con_weight * loss_con                            
                con_loss_val = loss_con.item()
            else:
                                                        
                loss = loss_seg
                con_loss_val = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
                                
            with torch.no_grad():
                  preds = torch.sigmoid(low_res_logits) > 0.5
                  mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                  preds_up = F.interpolate(preds.float(), size=(mask_h, mask_w), mode='nearest')
                  for i in range(masks.shape[0]):
                      pred_np = preds_up[i, 0].cpu().numpy() > 0.5
                      mask_i = masks[i][0] if masks[i].ndim == 3 else masks[i]
                      mask_np = mask_i.cpu().numpy() > 0.5
                      train_dices.append(compute_dice(pred_np, mask_np))
                      
            pbar.set_postfix({
                'L_all': f"{loss.item():.3f}",
                'L_con': f"{con_loss_val:.3f}",                               
                'dice': f"{np.mean(train_dices):.3f}"
            })
        
                                                              
        fusion_model.eval()
        val_losses = []
        val_dices = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar_val:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                bboxes = batch['bbox'].to(device)
                text_embeds = batch['text_embed'].to(device)
                
                if use_precomputed and 'image_embedding' in batch:
                    img_embeddings = batch['image_embedding'].to(device)
                else:
                    img_embeddings = medsam_model.image_encoder(images)
                            
                fusion_output = fusion_model(img_embeddings, text_embeds)
                if isinstance(fusion_output, tuple):
                    enhanced_embeddings = fusion_output[0]
                else:
                    enhanced_embeddings = fusion_output
                
                sparse_emb, dense_emb = medsam_model.prompt_encoder(
                    points=None, boxes=bboxes[:, None, :], masks=None
                )
                
                low_res_logits, _ = medsam_model.mask_decoder(
                    image_embeddings=enhanced_embeddings,
                    image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                
                masks = normalize_mask_for_loss(masks)
                masks_resized = F.interpolate(masks, size=low_res_logits.shape[-2:], mode='nearest')
                
                                                         
                loss_val = seg_criterion(low_res_logits, masks_resized)
                val_losses.append(loss_val.item())
                
                preds = torch.sigmoid(low_res_logits) > 0.5
                mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                preds_up = F.interpolate(preds.float(), size=(mask_h, mask_w), mode='nearest')

                for i in range(masks.shape[0]):
                    pred_np = preds_up[i, 0].cpu().numpy() > 0.5
                    mask_i = masks[i][0] if masks[i].ndim == 3 else masks[i]
                    mask_np = mask_i.cpu().numpy() > 0.5
                    val_dices.append(compute_dice(pred_np, mask_np))
                
                pbar_val.set_postfix({'dice': f"{np.mean(val_dices):.3f}"})
        
                                                           
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
        
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'gate_value': fusion_model.gate.item() if hasattr(fusion_model, 'gate') else None
            }, save_path)

    return history