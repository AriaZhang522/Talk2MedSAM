                                                       
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
from data_loaders_quick import create_nsclc_dataloaders            
from train_quick import train_fusion_model
from evaluate_quick import (
    evaluate_baseline, 
    evaluate_text_guided, 
    evaluate_counterfactual,             
    compare_results
)
from utils_quick import precompute_text_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                                  
config.FUSION_VERSION = "v4"    
config.BATCH_SIZE = 4         
config.NUM_EPOCHS = 20       
config.LEARNING_RATE = 1e-4

with suppress_stderr():  
    medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)
    tokenizer, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)

                                   
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

train_loader, val_loader = create_nsclc_dataloaders(
    csv_path=config.NSCLC_CSV_PATH,
    data_root=config.NSCLC_DATA_ROOT,
    text_embeddings_dict=text_embeddings_dict,
    train_split=0.8,
    batch_size=config.BATCH_SIZE
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

                                                          
model_save_path = f"{config.RESULTS_DIR}/{config.MODEL_NAME}_nsclc_{config.FUSION_VERSION}_FULL_best.pth"

                           
SKIP_TRAINING = False                  

if not SKIP_TRAINING:

    
    history = train_fusion_model(
        medsam_model=medsam_model,
        fusion_model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=device,
        save_path=model_save_path,
        use_precomputed=False                                
    )
    print(f"\nTraining completed! Best model saved.")
else:
    print("\n skipping training... Will load existing model for evaluation.")

   
print("Loading best model for evaluation...")
try:
    checkpoint = torch.load(model_save_path, weights_only=False, map_location=device)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    fusion_model.eval()
    print(f"Loaded model from {model_save_path}")
except FileNotFoundError:

    print("   Please train the model first by setting SKIP_TRAINING = False")
    sys.exit(1)


baseline_save_path = f"{config.RESULTS_DIR}/nsclc_{config.FUSION_VERSION}_baseline_results.csv"
baseline_results, baseline_dice = evaluate_baseline(
    medsam_model=medsam_model,
    data_loader=val_loader,
    device=device,
    save_path=baseline_save_path
)
print(f"Baseline Mean Dice: {baseline_dice:.4f}")


v2_model_path = f"{config.RESULTS_DIR}/{config.MODEL_NAME}_nsclc_best.pth"              
try:
    fusion_model_v2 = load_fusion_model(
        version="v2",
        save_path=v2_model_path,
        device=device
    )
    fusion_model_v2.eval()
    
    v2_save_path = f"{config.RESULTS_DIR}/nsclc_v2_text_guided_results.csv"
    v2_results, v2_dice = evaluate_text_guided(
        medsam_model=medsam_model,
        fusion_model=fusion_model_v2,
        data_loader=val_loader,
        device=device,
        save_path=v2_save_path
    )
    print(f"V2 Text-Guided Mean Dice: {v2_dice:.4f}")
except FileNotFoundError:
    v2_results = None
    v2_dice = None

v4_save_path = f"{config.RESULTS_DIR}/nsclc_{config.FUSION_VERSION}_text_guided_results.csv"
v4_results, v4_dice = evaluate_text_guided(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    save_path=v4_save_path
)


from scipy import stats

                 
baseline_dice_list = [r['dice'] for r in baseline_results]
v4_dice_list = [r['dice'] for r in v4_results]

          
assert len(baseline_dice_list) == len(v4_dice_list), "Sample counts don't match!"

                   
t_stat_v4, p_value_v4 = stats.ttest_rel(v4_dice_list, baseline_dice_list)
print(f"\n1. V4 vs Baseline:")
print(f"   t-statistic: {t_stat_v4:.4f}")
print(f"   p-value: {p_value_v4:.4e}")

                        
if v2_results is not None:
    v2_dice_list = [r['dice'] for r in v2_results]
    assert len(v4_dice_list) == len(v2_dice_list), "V2 and V4 sample counts don't match!"
    
    t_stat_v4_v2, p_value_v4_v2 = stats.ttest_rel(v4_dice_list, v2_dice_list)
    print(f"\n2. V4 vs V2:")
    print(f"   t-statistic: {t_stat_v4_v2:.4f}")
    print(f"   p-value: {p_value_v4_v2:.4e}")

    
            
    stats_results = pd.DataFrame({
        'Comparison': ['V4 vs Baseline', 'V4 vs V2'],
        't_statistic': [t_stat_v4, t_stat_v4_v2],
        'p_value': [p_value_v4, p_value_v4_v2],
        'significant_at_0.05': [p_value_v4 < 0.05, p_value_v4_v2 < 0.05]
    })
else:
                        
    stats_results = pd.DataFrame({
        'Comparison': ['V4 vs Baseline'],
        't_statistic': [t_stat_v4],
        'p_value': [p_value_v4],
        'significant_at_0.05': [p_value_v4 < 0.05]
    })

stats_save_path = f"{config.RESULTS_DIR}/nsclc_{config.FUSION_VERSION}_statistical_tests.csv"
stats_results.to_csv(stats_save_path, index=False)


                 
text_embeddings_dict_with_cf = {
    't_stage': t_stage_embeddings,
    'histology': histology_embeddings,
    't_stage_cf_map': config.T_STAGE_COUNTERFACTUAL,
    'histology_cf_map': config.HISTOLOGY_COUNTERFACTUAL
}


cf_t_save_path = f"{config.RESULTS_DIR}/nsclc_{config.FUSION_VERSION}_counterfactual_t_stage.csv"
cf_t_results, cf_t_dice = evaluate_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    text_embeddings_dict=text_embeddings_dict_with_cf,
    modify_t_stage=True,
    modify_histology=False,
    save_path=cf_t_save_path
)

cf_h_save_path = f"{config.RESULTS_DIR}/nsclc_{config.FUSION_VERSION}_counterfactual_histology.csv"
cf_h_results, cf_h_dice = evaluate_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    text_embeddings_dict=text_embeddings_dict_with_cf,
    modify_t_stage=False,
    modify_histology=True,
    save_path=cf_h_save_path
)

cf_both_save_path = f"{config.RESULTS_DIR}/nsclc_{config.FUSION_VERSION}_counterfactual_both.csv"
cf_both_results, cf_both_dice = evaluate_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    text_embeddings_dict=text_embeddings_dict_with_cf,
    modify_t_stage=True,
    modify_histology=True,
    save_path=cf_both_save_path
)

print(f"Baseline Dice:           {baseline_dice:.4f}")
if v2_dice is not None:
    print(f"V2 Text-Guided Dice:     {v2_dice:.4f} (Δ +{v2_dice - baseline_dice:.4f})")
print(f"V4 Text-Guided Dice:     {v4_dice:.4f} (Δ +{v4_dice - baseline_dice:.4f})")
print(f"\nCounterfactual Results:")
print(f"  Wrong T-stage:         {cf_t_dice:.4f} (Drop: {v4_dice - cf_t_dice:.4f})")
print(f"  Wrong Histology:       {cf_h_dice:.4f} (Drop: {v4_dice - cf_h_dice:.4f})")
print(f"  Both Wrong:            {cf_both_dice:.4f} (Drop: {v4_dice - cf_both_dice:.4f})")
