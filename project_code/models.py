"""
Model definitions for Text-Guided MedSAM

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Version 1: Basic Text-Guided Fusion
# ============================================================
class TextGuidedFusionV1(nn.Module):

    def __init__(self, visual_dim=256, text_dim=768, hidden_dim=256):
        super().__init__()
        # Project text embedding to visual dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim)
        )

        # Cross-attention
        self.query = nn.Linear(visual_dim, hidden_dim)
        self.key = nn.Linear(visual_dim, hidden_dim)
        self.value = nn.Linear(visual_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, visual_dim)

        self.scale = hidden_dim ** -0.5
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, visual_feat, text_embed):
        """
        visual_feat: [B, C, H, W] - image features from MedSAM encoder
        text_embed: [B, 768] - text embedding from PubMedBERT
        """
        B, C, H, W = visual_feat.shape

        # Project text
        text_proj = self.text_proj(text_embed)  # [B, visual_dim]
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Flatten visual features
        visual_flat = visual_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # Expand text for attention
        text_key = text_proj.flatten(2).permute(0, 2, 1)  # [B, 1, C]

        # Compute attention
        Q = self.query(visual_flat)  # [B, HW, hidden]
        K = self.key(text_key)       # [B, 1, hidden]
        V = self.value(text_key)     # [B, 1, hidden]

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, HW, 1]
        attn = F.softmax(attn, dim=-1)

        out = attn @ V  # [B, HW, hidden]
        out = self.out_proj(out)  # [B, HW, visual_dim]

        # Residual connection
        out = self.norm(visual_flat + out)

        # Reshape back
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out


class TextGuidedFusionV2(nn.Module):

    def __init__(self, visual_dim=256, text_dim=768, hidden_dim=256):
        super().__init__()
        # Project text embedding to visual dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, visual_dim)
        )

        # Cross-attention
        self.query = nn.Linear(visual_dim, hidden_dim)
        self.key = nn.Linear(visual_dim, hidden_dim)
        self.value = nn.Linear(visual_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, visual_dim)

        self.scale = hidden_dim ** -0.5
        self.norm = nn.LayerNorm(visual_dim)

        # Learnable gate - initialized to give more weight to text
        self.gate = nn.Parameter(torch.tensor(1.0))  # Start with higher text influence

    def forward(self, visual_feat, text_embed):
        """
        visual_feat: [B, C, H, W]
        text_embed: [B, 768]
        """
        B, C, H, W = visual_feat.shape

        # Project text
        text_proj = self.text_proj(text_embed)  # [B, visual_dim]
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Flatten visual features
        visual_flat = visual_feat.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # Expand text for attention
        text_key = text_proj.flatten(2).permute(0, 2, 1)  # [B, 1, C]

        # Compute attention
        Q = self.query(visual_flat)
        K = self.key(text_key)
        V = self.value(text_key)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ V
        out = self.out_proj(out)

        # Apply learnable gate (sigmoid to keep it in [0, 1])
        gate = torch.sigmoid(self.gate)
        
        # Residual with gated text influence
        out = self.norm(visual_flat + gate * out)

        # Reshape back
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out


# ============================================================
# Factory function to create fusion module
# ============================================================
def create_fusion_model(version="v2", visual_dim=256, text_dim=768, hidden_dim=256):

    if version == "v1":
        return TextGuidedFusionV1(visual_dim, text_dim, hidden_dim)
    elif version == "v2":
        return TextGuidedFusionV2(visual_dim, text_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown fusion version: {version}")


# ============================================================
# Model loading utilities
# ============================================================
def load_medsam(checkpoint_path, device):

    from MedSAM.segment_anything import sam_model_registry
    
    model_type = "vit_b"
    medsam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    medsam_model.to(device)
    medsam_model.eval()
    
    # Freeze MedSAM parameters
    for param in medsam_model.parameters():
        param.requires_grad = False
    
    return medsam_model


def load_text_encoder(model_name, device):

    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder = AutoModel.from_pretrained(model_name)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    
    # Freeze text encoder parameters
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    return tokenizer, text_encoder


def save_fusion_model(fusion_model, save_path):

    torch.save({
        'model_state_dict': fusion_model.state_dict(),
        'gate_value': fusion_model.gate.item() if hasattr(fusion_model, 'gate') else None
    }, save_path)
    print(f"Model saved to: {save_path}")



def load_fusion_model(version, save_path, visual_dim=256, text_dim=768, hidden_dim=256, device='cuda'):

    fusion_model = create_fusion_model(version, visual_dim, text_dim, hidden_dim)
    
    # checkpoint = torch.load(save_path, map_location=device)
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:

        state_dict = checkpoint

    fusion_model.load_state_dict(state_dict)
    fusion_model.to(device)


    if isinstance(checkpoint, dict) and 'gate_value' in checkpoint and checkpoint['gate_value'] is not None:
        print(f"Loaded model with gate value: {checkpoint['gate_value']:.4f}")
    
    return fusion_model