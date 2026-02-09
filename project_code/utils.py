"""
Utility functions for Text-Guided MedSAM
"""
import torch
import numpy as np
from PIL import Image
from skimage import transform
import torch.nn.functional as F



def get_bbox_from_mask(mask):

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return np.array([x_min, y_min, x_max, y_max])


def resize_image_and_bbox(image, bbox, target_size=1024):

    H, W = image.shape[:2]
    
    # Resize image
    image_resized = transform.resize(
        image, 
        (target_size, target_size), 
        preserve_range=True, 
        anti_aliasing=True
    ).astype(np.uint8)
    
    # Scale bbox
    scale_x, scale_y = target_size / W, target_size / H
    bbox_resized = np.array([
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y
    ])
    
    return image_resized, bbox_resized


def resize_mask(mask, original_size):

    mask_resized = transform.resize(
        mask.astype(float),
        original_size,
        order=0,  # nearest neighbor
        preserve_range=True,
        anti_aliasing=False
    )
    return mask_resized > 0.5


# ============================================================
# Evaluation metrics
# ============================================================
def compute_dice(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / union
    return dice


def compute_iou(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


# ============================================================
# Text encoding utilities
# ============================================================
@torch.no_grad()
def encode_text_prompt(text, tokenizer, text_encoder, device):

    if isinstance(text, str):
        text = [text]
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    # Encode
    outputs = text_encoder(**inputs)
    # Use [CLS] token embedding
    text_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
    
    return text_embedding


def precompute_text_embeddings(prompts_dict, tokenizer, text_encoder, device):

    embeddings_dict = {}
    
    for category, prompt in prompts_dict.items():
        embedding = encode_text_prompt(prompt, tokenizer, text_encoder, device)
        embedding = embedding.squeeze(0)
        embeddings_dict[category] = embedding.cpu()  # Store on CPU to save GPU memory
    
    return embeddings_dict


# ============================================================
# MedSAM inference utilities
# ============================================================
@torch.no_grad()
def medsam_inference(medsam_model, image_embedding, bbox_1024, H, W):

    device = image_embedding.device
    
    # Prepare bbox for prompt encoder
    bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float).unsqueeze(0).to(device)
    
    # Get prompt embeddings
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=bbox_torch,
        masks=None,
    )
    
    # Predict mask
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    
    # Upsample to 1024x1024
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    
    # Threshold and resize to original size
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    pred_mask = (low_res_pred > 0.5).astype(np.uint8)
    pred_mask = resize_mask(pred_mask, (H, W))
    
    return pred_mask


@torch.no_grad()
def text_guided_inference(medsam_model, fusion_model, image_embedding, text_embedding, 
                         bbox_1024, H, W):

    device = image_embedding.device
    
    # Fuse image and text features
    enhanced_embedding = fusion_model(image_embedding, text_embedding)
    
    # Prepare bbox for prompt encoder
    bbox_torch = torch.as_tensor(bbox_1024, dtype=torch.float).unsqueeze(0).to(device)
    
    # Get prompt embeddings
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=bbox_torch,
        masks=None,
    )
    
    # Predict mask using enhanced features
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=enhanced_embedding,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    
    # Upsample to 1024x1024
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    
    # Threshold and resize to original size
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    pred_mask = (low_res_pred > 0.5).astype(np.uint8)
    pred_mask = resize_mask(pred_mask, (H, W))
    
    return pred_mask


# ============================================================
# Result saving utilities
# ============================================================
def save_results_to_csv(results_list, save_path):

    import pandas as pd
    
    df = pd.DataFrame(results_list)
    df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Mean Dice: {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    if 'iou' in df.columns:
        print(f"  Mean IoU:  {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")


# ============================================================
# Visualization utilities
# ============================================================
def visualize_prediction(image, gt_mask, pred_mask, title="Prediction"):

    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Blues')
    dice = compute_dice(pred_mask, gt_mask)
    axes[2].set_title(f"Prediction (Dice: {dice:.4f})")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 