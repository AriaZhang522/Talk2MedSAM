import sys
import os
import contextlib
import warnings
import torch
import pandas as pd
import numpy as np

         
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
    create_fusion_model, 
    save_fusion_model, 
    load_fusion_model
)
from data_loaders_quick import (
    create_bus_dataloaders,                  
    precompute_bus_image_embeddings          
)
from train_quick import train_fusion_model
from evaluate_quick import (
    evaluate_baseline, 
    evaluate_text_guided, 
    evaluate_bus_counterfactual, 
    compare_results
)
from utils_quick import precompute_text_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
config.FUSION_VERSION = "v4"            
config.BATCH_SIZE =  4                                          
config.NUM_EPOCHS = 20       
config.LEARNING_RATE = 1e-4


with suppress_stderr():  
    medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)
    tokenizer, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)


birads_embeddings = precompute_text_embeddings(
    config.BIRADS_PROMPTS, tokenizer, text_encoder, device
)

                                    
                               
embeddings_save_path = f"{config.RESULTS_DIR}/bus_image_embeddings.pt"

print("   Checking/Computing full dataset image embeddings...")
if os.path.exists(embeddings_save_path):
    print(f"   ✓ Loading cached embeddings from {embeddings_save_path}")
    img_embeddings_dict = torch.load(embeddings_save_path)
else:
    print("   Creating embeddings (this may take a few minutes)...")
    img_embeddings_dict = precompute_bus_image_embeddings(
        medsam_model=medsam_model,
        csv_path=config.BUS_CSV_PATH,              
        base_path=config.BUS_DATA_ROOT,
        text_embeddings_dict=birads_embeddings,                         
        save_path=embeddings_save_path,
        device=device
    )

                                                                        
train_loader, val_loader = create_bus_dataloaders(
    csv_path=config.BUS_CSV_PATH,
    base_path=config.BUS_DATA_ROOT,
    text_embeddings_dict=birads_embeddings,
    train_split=0.8,                                
    batch_size=config.BATCH_SIZE,
    precomputed_img_embeddings=img_embeddings_dict          
)



                                                              
                                  
                                                              
print(f"\n ({config.FUSION_VERSION})...")
fusion_model = create_fusion_model(
    version=config.FUSION_VERSION,
    visual_dim=config.VISUAL_DIM,
    text_dim=config.TEXT_DIM,
    hidden_dim=config.HIDDEN_DIM
)
fusion_model.to(device)
print(f"   Trainable params: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")
  
model_save_path = f"{config.RESULTS_DIR}/{config.MODEL_NAME}_bus_{config.FUSION_VERSION}_FULL_best.pth"
print(f"Best model will be saved to: {model_save_path}")

            
print("Loading best model for evaluation...")
checkpoint = torch.load(model_save_path, weights_only=False) 
fusion_model.load_state_dict(checkpoint['model_state_dict'])
                                                                               
fusion_model.eval()

                            
save_path_tg = f"{config.RESULTS_DIR}/full_{config.FUSION_VERSION}_results.csv"
tg_results, tg_dice = evaluate_text_guided(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    save_path=save_path_tg
)
print(f"Text-Guided Mean Dice: {tg_dice:.4f}")

                                              
print("\nRunning Counterfactual Check on Full Validation Set...")
save_path_cf = f"{config.RESULTS_DIR}/full_{config.FUSION_VERSION}_counterfactual.csv"
cf_results, cf_dice = evaluate_bus_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    birads_embeddings=birads_embeddings,
    counterfactual_map=config.BIRADS_COUNTERFACTUAL,
    use_wrong_birads=True,
    save_path=save_path_cf
)

print(f"\nCorrect Dice: {tg_dice:.4f}")
print(f"Wrong Dice:   {cf_dice:.4f}")
print(f"Drop:         {tg_dice - cf_dice:.4f}")
