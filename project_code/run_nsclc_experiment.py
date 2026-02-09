

# ============================================================
# Step 1: Setup and Imports
# ============================================================
import sys
sys.path.append('/content/drive/MyDrive/Signal_Final_Project')
sys.path.append('/content/drive/MyDrive/Signal_Final_Project/MedSAM')
import torch
import config
from models import load_medsam, load_text_encoder, create_fusion_model, save_fusion_model, load_fusion_model
from data_loaders import create_nsclc_dataloaders
from train import train_fusion_model
from evaluate import evaluate_baseline, evaluate_text_guided, evaluate_counterfactual, compare_results
from utils import precompute_text_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)

tokenizer, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)

t_stage_embeddings = precompute_text_embeddings(
    config.T_STAGE_PROMPTS,
    tokenizer,
    text_encoder,
    device
)

histology_embeddings = precompute_text_embeddings(
    config.HISTOLOGY_PROMPTS,
    tokenizer,
    text_encoder,
    device
)

text_embeddings_dict = {
    't_stage': t_stage_embeddings,
    'histology': histology_embeddings,
    # For counterfactual experiments
    't_stage_cf_map': config.T_STAGE_COUNTERFACTUAL,
    'histology_cf_map': config.HISTOLOGY_COUNTERFACTUAL
}

train_loader, val_loader = create_nsclc_dataloaders(
    csv_path=config.NSCLC_CSV_PATH,
    data_root=config.NSCLC_DATA_ROOT,
    text_embeddings_dict=text_embeddings_dict,
    train_split=config.TRAIN_SPLIT,
    batch_size=config.BATCH_SIZE
)


fusion_model = create_fusion_model(
    version=config.FUSION_VERSION,
    visual_dim=config.VISUAL_DIM,
    text_dim=config.TEXT_DIM,
    hidden_dim=config.HIDDEN_DIM
)
fusion_model.to(device)

print(f"  Total trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")



model_save_path = config.get_model_save_path(dataset_name="nsclc")
print(f"Model will be saved to: {model_save_path}")

history = train_fusion_model(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=config.NUM_EPOCHS,
    learning_rate=config.LEARNING_RATE,
    device=device,
    save_path=model_save_path
)


baseline_save_path = config.get_results_save_path("baseline", dataset_name="nsclc")
baseline_results, baseline_dice = evaluate_baseline(
    medsam_model=medsam_model,
    data_loader=val_loader,
    device=device,
    save_path=baseline_save_path
)

print(f"\nBaseline Mean Dice: {baseline_dice:.4f}")



text_guided_save_path = config.get_results_save_path("text_guided", dataset_name="nsclc")
text_guided_results, text_guided_dice = evaluate_text_guided(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    save_path=text_guided_save_path
)

print(f"\nText-Guided Mean Dice: {text_guided_dice:.4f}")

compare_results(baseline_results, text_guided_results, dataset_name="NSCLC Dataset")


print("COUNTERFACTUAL EXPERIMENT 1: WRONG T-STAGE")
# print("="*70)
print("Purpose: Test if model relies on T-stage information")
print("Method: Use incorrect T-stage prompts (e.g., T1a → T4)")

cf_t_save_path = config.get_results_save_path("counterfactual_t", dataset_name="nsclc")
cf_t_results, cf_t_dice = evaluate_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    text_embeddings_dict=text_embeddings_dict,
    modify_t_stage=True,
    modify_histology=False,
    save_path=cf_t_save_path
)

print(f"\nWrong T-Stage Mean Dice: {cf_t_dice:.4f}")
print(f"Performance Drop: {text_guided_dice - cf_t_dice:.4f}")
print(f"{'Model DOES rely on T-stage info ✓' if cf_t_dice < text_guided_dice - 0.01 else 'Model may NOT use T-stage effectively ✗'}")

print("COUNTERFACTUAL EXPERIMENT 2: WRONG HISTOLOGY")

print("Purpose: Test if model relies on histology information")
print("Method: Use incorrect histology prompts (Adenocarcinoma ↔ Squamous)")

cf_h_save_path = config.get_results_save_path("counterfactual_h", dataset_name="nsclc")
cf_h_results, cf_h_dice = evaluate_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    text_embeddings_dict=text_embeddings_dict,
    modify_t_stage=False,
    modify_histology=True,
    save_path=cf_h_save_path
)

print(f"\nWrong Histology Mean Dice: {cf_h_dice:.4f}")
print(f"Performance Drop: {text_guided_dice - cf_h_dice:.4f}")
print(f"→ {'Model DOES rely on histology info ✓' if cf_h_dice < text_guided_dice - 0.01 else 'Model may NOT use histology effectively ✗'}")


print("COUNTERFACTUAL EXPERIMENT 3: WRONG T-STAGE + HISTOLOGY")

print("Purpose: Test combined effect of all wrong prompts")
print("Method: Use incorrect T-stage AND histology prompts")

cf_both_save_path = config.get_results_save_path("counterfactual_both", dataset_name="nsclc")
cf_both_results, cf_both_dice = evaluate_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    text_embeddings_dict=text_embeddings_dict,
    modify_t_stage=True,
    modify_histology=True,
    save_path=cf_both_save_path
)

print(f"\nWrong Both Mean Dice: {cf_both_dice:.4f}")
print(f"Performance Drop: {text_guided_dice - cf_both_dice:.4f}")
print(f"→ {'Strongest performance drop - model truly uses text! ✓' if cf_both_dice < cf_t_dice and cf_both_dice < cf_h_dice else 'Mixed results'}")


import pandas as pd
summary = pd.DataFrame({
    'Experiment': [
        'Baseline (No Text)',
        'Text-Guided (Correct)',
        'Counterfactual: Wrong T-Stage',
        'Counterfactual: Wrong Histology',
        'Counterfactual: Wrong Both'
    ],
    'Mean Dice': [
        baseline_dice,
        text_guided_dice,
        cf_t_dice,
        cf_h_dice,
        cf_both_dice
    ],
    'vs Baseline': [
        0,
        text_guided_dice - baseline_dice,
        cf_t_dice - baseline_dice,
        cf_h_dice - baseline_dice,
        cf_both_dice - baseline_dice
    ],
    'vs Text-Guided': [
        baseline_dice - text_guided_dice,
        0,
        cf_t_dice - text_guided_dice,
        cf_h_dice - text_guided_dice,
        cf_both_dice - text_guided_dice
    ]
})

print(summary.to_string(index=False))