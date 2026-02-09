
import sys
import os
import contextlib

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@contextlib.contextmanager
def suppress_stderr():
    """
    Silences C++/CUDA/TF logs that write directly to file descriptor 2 (stderr).
    """

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
import warnings
warnings.filterwarnings('ignore')
import torch
import config
from models import load_medsam, load_text_encoder, create_fusion_model, save_fusion_model, load_fusion_model
from data_loaders import create_bus_dataloaders, precompute_bus_image_embeddings
from train import train_fusion_model
from evaluate import evaluate_baseline, evaluate_text_guided, compare_results, evaluate_bus_counterfactual
# from evaluate import evaluate_baseline, evaluate_text_guided, compare_results, evaluate_bus_counterfactual
from utils import precompute_text_embeddings
from train import debug_val_one_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Loading MedSAM...")
with suppress_stderr():  
  medsam_model = load_medsam(config.MEDSAM_CHECKPOINT, device)

with suppress_stderr():  
  tokenizer, text_encoder = load_text_encoder(config.TEXT_ENCODER_MODEL, device)

birads_embeddings = precompute_text_embeddings(
    config.BIRADS_PROMPTS,
    tokenizer,
    text_encoder,
    device
)


embeddings_save_path = f"{config.RESULTS_DIR}/bus_image_embeddings.pt"


img_embeddings_dict = precompute_bus_image_embeddings(
    medsam_model=medsam_model,
    csv_path=config.BUS_CSV_PATH,
    base_path=config.BUS_DATA_ROOT,
    text_embeddings_dict=birads_embeddings,
    save_path=embeddings_save_path,
    device=device
)



print("\nCreating data loaders...")
train_loader, val_loader = create_bus_dataloaders(
    csv_path=config.BUS_CSV_PATH,
    base_path=config.BUS_DATA_ROOT,
    text_embeddings_dict=birads_embeddings,
    train_split=config.TRAIN_SPLIT,
    batch_size=config.BATCH_SIZE,
    precomputed_img_embeddings=img_embeddings_dict
)


def precompute_bus_image_embeddings(medsam_model, data_loader, device, save_path):

    medsam_model.eval()
    all_embs = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Precomputing BUS image embeddings"):
            images = batch['image'].to(device)       # [B, 3, 1024, 1024]
            img_ids = batch['image_id']              # list of strings

            img_embs = medsam_model.image_encoder(images)  # [B, C, H, W]

            for img_id, emb in zip(img_ids, img_embs):
                all_embs[img_id] = emb.cpu()

    torch.save(all_embs, save_path)


fusion_model = create_fusion_model(
    version=config.FUSION_VERSION,
    visual_dim=config.VISUAL_DIM,
    text_dim=config.TEXT_DIM,
    hidden_dim=config.HIDDEN_DIM
)
fusion_model.to(device)
print(f"Fusion model created (version: {config.FUSION_VERSION})")
print(f"  Total trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")

# ============================================================
# Step 6: Train Fusion Module
# ============================================================
# print("\n" + "="*70)
# print("TRAINING")
# print("="*70)

# model_save_path = config.get_model_save_path(dataset_name="bus")
# print(f"Model will be saved to: {model_save_path}")

# history = train_fusion_model(
#     medsam_model=medsam_model,
#     fusion_model=fusion_model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     num_epochs=config.NUM_EPOCHS,
#     learning_rate=config.LEARNING_RATE,
#     device=device,
#     save_path=model_save_path,
#     use_precomputed=True 
# )

# print("\nTraining completed!")


# print("\n[DEBUG] Running only one VAL batch on CPU to check pipeline...")
# debug_val_one_batch(
#     medsam_model=medsam_model,
#     fusion_model=fusion_model,
#     val_loader=val_loader,
#     device=device
# )



print("Loading best model...")
fusion_model = load_fusion_model(
    version=config.FUSION_VERSION,
    save_path=config.get_model_save_path(dataset_name="bus"),
    device=device
)



baseline_save_path = config.get_results_save_path("baseline", dataset_name="bus")
baseline_results, baseline_dice = evaluate_baseline(
    medsam_model=medsam_model,
    data_loader=val_loader,
    device=device,
    save_path=baseline_save_path
)

print(f"\nBaseline Mean Dice: {baseline_dice:.4f}")


print("EVALUATION: TEXT-GUIDED")


text_guided_save_path = config.get_results_save_path("text_guided", dataset_name="bus")
text_guided_results, text_guided_dice = evaluate_text_guided(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    save_path=text_guided_save_path
)

print(f"\nText-Guided Mean Dice: {text_guided_dice:.4f}")


compare_results(baseline_results, text_guided_results, dataset_name="BUS Dataset")

print("EXPERIMENT COMPLETED!")
# print("="*70)
print(f"\nResults saved in: {config.RESULTS_DIR}")
print(f"  - Model: {config.get_model_save_path(dataset_name='bus')}")
print(f"  - Baseline results: {baseline_save_path}")
print(f"  - Text-guided results: {text_guided_save_path}")


# Import the counterfactual evaluation function
from evaluate import evaluate_bus_counterfactual

# Load the counterfactual mapping from config
print(f"Counterfactual mapping: {config.BIRADS_COUNTERFACTUAL}")

# Experiment 1: Correct BI-RADS (for comparison)
print("\nExperiment 1: Correct BI-RADS")
correct_cf_save_path = config.get_results_save_path("counterfactual_correct", dataset_name="bus")
correct_cf_results, correct_cf_dice = evaluate_text_guided(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    save_path=correct_cf_save_path
)
print(f"Correct BI-RADS Mean Dice: {correct_cf_dice:.4f}")


# Experiment 2: Wrong BI-RADS (counterfactual)
print("\nExperiment 2: Wrong BI-RADS")
wrong_cf_save_path = config.get_results_save_path("counterfactual_wrong", dataset_name="bus")
wrong_cf_results, wrong_cf_dice = evaluate_bus_counterfactual(
    medsam_model=medsam_model,
    fusion_model=fusion_model,
    data_loader=val_loader,
    device=device,
    birads_embeddings=birads_embeddings,
    counterfactual_map=config.BIRADS_COUNTERFACTUAL,
    use_wrong_birads=True,
    save_path=wrong_cf_save_path
)
print(f"Wrong BI-RADS Mean Dice: {wrong_cf_dice:.4f}")


print("COUNTERFACTUAL ANALYSIS")

print(f"Correct BI-RADS:  {correct_cf_dice:.4f}")
print(f"Wrong BI-RADS:    {wrong_cf_dice:.4f}")
performance_drop = correct_cf_dice - wrong_cf_dice
print(f"Performance Drop: {performance_drop:.4f} ({performance_drop/correct_cf_dice*100:.2f}%)")


if performance_drop > 0.01:
    print("\n Model IS using text information! (Performance drops with wrong BI-RADS)")
else:
    print("\n Model may NOT be using text information effectively.")

print(f"\nCounterfactual results saved:")
print(f"  - Correct BI-RADS: {correct_cf_save_path}")
print(f"  - Wrong BI-RADS: {wrong_cf_save_path}")


print(f"\nResults saved in: {config.RESULTS_DIR}")
print(f"  - Model: {config.get_model_save_path(dataset_name='bus')}")
print(f"  - Baseline results: {baseline_save_path}")
print(f"  - Text-guided results: {text_guided_save_path}")
print(f"  - Counterfactual (correct): {correct_cf_save_path}")
print(f"  - Counterfactual (wrong): {wrong_cf_save_path}")