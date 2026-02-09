"""
Dataset loaders for BUS and NSCLC datasets

"""
import sys
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    
from utils import get_bbox_from_mask, resize_image_and_bbox


# ============================================================
# BUS (Breast Ultrasound) Dataset
# ============================================================
IMG_SIZE = (256, 256) 
class BUSDataset(Dataset):
    """
    BUS Dataset with BI-RADS text prompts
    """
    def __init__(self, df, base_path, text_embeddings_dict,precomputed_img_embeddings=None):
        """
        Args:
            df: DataFrame with columns ['ID', 'BIRADS']
            base_path: root directory with Images/ and Masks/ folders
            text_embeddings_dict: dict mapping BIRADS category -> text embedding
        """
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.text_embeddings_dict = text_embeddings_dict
        self.precomputed_img_embeddings = precomputed_img_embeddings

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['ID']
        birads = row['BIRADS']

        image_path = f'{self.base_path}/Images/{img_id}.png'
        mask_path  = f'{self.base_path}/Masks/mask_{img_id[4:]}.png'
        # mask_path = f'{base_path}/Masks/mask_{img_id[4:]}.png'


        image = np.array(Image.open(image_path).convert('RGB'))          # [H,W,3]
        gt_mask = np.array(Image.open(mask_path).convert('L')) > 0       # [H,W] bool
        
        H, W = image.shape[:2]

        bbox = get_bbox_from_mask(gt_mask)
        if bbox is None:

            bbox = np.array([0, 0, W - 1, H - 1], dtype=np.float32)
        else:
            bbox = np.array(bbox, dtype=np.float32)


        image_1024, bbox_1024 = resize_image_and_bbox(image, bbox, target_size=1024)

        mask_img = Image.fromarray(gt_mask.astype(np.uint8) * 255)   # 0/1 -> 0/255
        mask_1024 = np.array(mask_img.resize((1024, 1024), Image.NEAREST)) > 0  # [H,W] bool


        image_tensor = torch.tensor(image_1024).permute(2, 0, 1).float()      # [3, 1024, 1024]
        mask_tensor = torch.tensor(mask_1024, dtype=torch.float32).unsqueeze(0)  # [1, 1024, 1024]
        bbox_tensor = torch.tensor(bbox_1024, dtype=torch.float32)            # [4]

        text_embed = self.text_embeddings_dict[birads]
        if isinstance(text_embed, np.ndarray):
            text_embed = torch.from_numpy(text_embed).float()
    
        result =  {
              'image': image_tensor,
              'mask': mask_tensor,
              'bbox': bbox_tensor,
              'text_embed': text_embed,
              'birads': birads,
              'image_id': img_id,
              'original_size': (H, W)
          }
        if self.precomputed_img_embeddings is not None:
            result['image_embedding'] = self.precomputed_img_embeddings[img_id]

        return result

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     img_id = row['ID']
    #     birads = row['BIRADS']

    #     # Load image and mask
    #     image_path = f'{self.base_path}/Images/{img_id}.png'
    #     mask_path = f'{self.base_path}/Masks/mask_{img_id[4:]}.png'

    #     image = np.array(Image.open(image_path).convert('RGB'))
    #     gt_mask = np.array(Image.open(mask_path).convert('L')) > 0

    #     H, W = image.shape[:2]

    #     # Get bbox from mask
    #     bbox = get_bbox_from_mask(gt_mask)

    #     image_resized, gt_mask_resized, bbox_resized = resize_image_and_bbox(
    #         image, gt_mask.astype(np.uint8), bbox, IMG_SIZE
    #     )

    #     image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    #     # mask: [1,H,W]
    #     mask_tensor = torch.from_numpy(gt_mask_resized.astype(np.float32)).unsqueeze(0)

        
    #     if bbox is None:
    #         # Handle empty mask
    #         bbox = np.array([0, 0, W-1, H-1])

    #     # Resize to 1024x1024
    #     image_1024, bbox_1024 = resize_image_and_bbox(image, bbox, target_size=1024)

    #     # Convert to tensor
    #     image_tensor = torch.tensor(image_1024).permute(2, 0, 1).float()  # [3, 1024, 1024]
    #     bbox_tensor = torch.tensor(bbox_1024, dtype=torch.float32)
    #     mask_tensor = torch.tensor(gt_mask, dtype=torch.float32)

    #     # Get text embedding
    #     text_embed = self.text_embeddings_dict[birads]  # [768]

    #     return {
    #         'image': image_tensor,
    #         'mask': mask_tensor,
    #         'bbox': bbox_tensor,
    #         'text_embed': text_embed,
    #         'birads': birads,
    #         'image_id': img_id,
    #         'original_size': (H, W)
    #     }


def create_bus_dataloaders(csv_path, base_path, text_embeddings_dict, 
                          train_split=0.8, batch_size=4, precomputed_img_embeddings=None):
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Total BUS samples: {len(df)}")
    
    # Split by BI-RADS category to ensure balanced split
    train_dfs = []
    val_dfs = []
    
    for birads in df['BIRADS'].unique():
        birads_df = df[df['BIRADS'] == birads]
        train_df, val_df = train_test_split(
            birads_df, 
            train_size=train_split, 
            random_state=42
        )
        train_dfs.append(train_df)
        val_dfs.append(val_df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = BUSDataset(train_df, base_path, text_embeddings_dict,precomputed_img_embeddings)
    val_dataset = BUSDataset(val_df, base_path, text_embeddings_dict,precomputed_img_embeddings)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader


@torch.no_grad()
def precompute_bus_image_embeddings(medsam_model, csv_path, base_path, 
                                    text_embeddings_dict, save_path, device):
    import os
    from tqdm import tqdm
    

    if os.path.exists(save_path):
        print(f"Loading pre-computed embeddings from {save_path}")
        return torch.load(save_path)
    
    print("Pre-computing image embeddings...")
    medsam_model.eval()
    
    df = pd.read_csv(csv_path)
    dataset = BUSDataset(df, base_path, text_embeddings_dict,precomputed_img_embeddings=None)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    embeddings_dict = {}
    
    for batch in tqdm(loader, desc="Computing embeddings"):
        images = batch['image'].to(device)
        image_ids = batch['image_id']
        
        # Compute embeddings
        img_embeddings = medsam_model.image_encoder(images)  # [B, 256, 64, 64]
        for img_id, emb in zip(image_ids, img_embeddings):
            embeddings_dict[img_id] = emb.cpu()
        
        # Store by image_id
        for img_id, emb in zip(image_ids, img_embeddings):
            embeddings_dict[img_id] = emb.cpu()
    
    # Save to disk
    torch.save(embeddings_dict, save_path)
    print(f"Saved embeddings to {save_path}")
    
    return embeddings_dict


# ============================================================
# NSCLC (Lung Cancer CT) Dataset
# ============================================================
class NSCLCDataset(Dataset):

    def __init__(self, df, data_root, text_embeddings_dict):
        

        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.text_embeddings_dict = text_embeddings_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = f"{self.data_root}/{row['ImagePath']}"
        mask_path = f"{self.data_root}/{row['MaskPath']}"
        
        # Load image and mask
        image = np.array(Image.open(image_path).convert('RGB'))
        gt_mask = np.array(Image.open(mask_path).convert('L')) > 0

        H, W = image.shape[:2]

        # Get bbox
        bbox = get_bbox_from_mask(gt_mask)
        
        if bbox is None:
            bbox = np.array([0, 0, W-1, H-1])

        # Resize to 1024x1024
        image_1024, bbox_1024 = resize_image_and_bbox(image, bbox, target_size=1024)

        # Convert to tensor
        image_tensor = torch.tensor(image_1024).permute(2, 0, 1).float()
        bbox_tensor = torch.tensor(bbox_1024, dtype=torch.float32)
        mask_tensor = torch.tensor(gt_mask, dtype=torch.float32)

        # Get text embeddings (we'll combine T-stage and histology)
        t_stage = row['T_Stage']
        histology = row['Histology']
        
        t_stage_embed = self.text_embeddings_dict['t_stage'][t_stage]
        histology_embed = self.text_embeddings_dict['histology'][histology]
        
        # Combine embeddings (simple average)
        combined_embed = (t_stage_embed + histology_embed) / 2

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'bbox': bbox_tensor,
            'text_embed': combined_embed,
            't_stage': t_stage,
            'histology': histology,
            'patient_id': row['PatientID'],
            'original_size': (H, W)
        }


def create_nsclc_dataloaders(csv_path, data_root, text_embeddings_dict,
                            train_split=0.8, batch_size=4):

    df = pd.read_csv(csv_path)
    
    # Filter out samples with empty masks
    valid_samples = []
    for idx, row in df.iterrows():
        mask_path = f"{data_root}/{row['MaskPath']}"
        try:
            mask = np.array(Image.open(mask_path).convert('L')) > 0
            if mask.sum() > 0:  # Has non-empty mask
                valid_samples.append(idx)
        except:
            continue
    
    df = df.loc[valid_samples].reset_index(drop=True)
    print(f"Total NSCLC samples with valid masks: {len(df)}")
    
    # Split by patient ID to avoid data leakage
    patient_ids = df['PatientID'].unique()
    train_patients, val_patients = train_test_split(
        patient_ids,
        train_size=train_split,
        random_state=42
    )
    
    train_df = df[df['PatientID'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['PatientID'].isin(val_patients)].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = NSCLCDataset(train_df, data_root, text_embeddings_dict)
    val_dataset = NSCLCDataset(val_df, data_root, text_embeddings_dict)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader


# ============================================================
# Counterfactual Dataset (for NSCLC experiments)
# ============================================================
def get_counterfactual_embedding(row, text_embeddings_dict, 
                                modify_t_stage=False, modify_histology=False):
    t_stage = row['T_Stage']
    histology = row['Histology']
    
    # Get correct or counterfactual T-stage
    if modify_t_stage and t_stage in text_embeddings_dict['t_stage_cf_map']:
        wrong_t_stage = text_embeddings_dict['t_stage_cf_map'][t_stage]
        t_embed = text_embeddings_dict['t_stage'][wrong_t_stage]
    else:
        t_embed = text_embeddings_dict['t_stage'][t_stage]
    
    # Get correct or counterfactual histology
    if modify_histology and histology in text_embeddings_dict['histology_cf_map']:
        wrong_histology = text_embeddings_dict['histology_cf_map'][histology]
        h_embed = text_embeddings_dict['histology'][wrong_histology]
    else:
        h_embed = text_embeddings_dict['histology'][histology]
    
    # Combine
    combined_embed = (t_embed + h_embed) / 2
    
    return combined_embed