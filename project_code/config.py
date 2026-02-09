"""
Configuration file for Text-Guided MedSAM experiments

"""

# ============================================================
# MODEL SETTINGS
# ============================================================
MODEL_NAME = "fusion_model"
FUSION_VERSION = "v2"  # "v1" or "v2" (v2 has learnable gate)

# ============================================================
# PATH SETTINGS
# ============================================================
# Google Drive root
DRIVE_ROOT = "/content/drive/MyDrive"
PROJECT_ROOT = f"{DRIVE_ROOT}/Signal_Final_Project"

# MedSAM paths
MEDSAM_DIR = f"{PROJECT_ROOT}/MedSAM"
MEDSAM_CHECKPOINT = f"{MEDSAM_DIR}/work_dir/MedSAM/medsam_vit_b.pth"

# BUS Dataset paths
BUS_DATA_ROOT = f"{DRIVE_ROOT}/BUS_BRA_data/BUSBRA/BUSBRA"
BUS_CSV_PATH = f"{DRIVE_ROOT}/BUS_BRA_data/BUSBRA/BUSBRA/bus_data.csv"

# NSCLC Dataset paths
NSCLC_DATA_ROOT = f"{DRIVE_ROOT}/NSCLC_Final_2.5D_Dataset_Verified"
NSCLC_CSV_PATH = f"{NSCLC_DATA_ROOT}/Final_Dataset_With_Clinical_Info.csv"

# Results save directory
RESULTS_DIR = f"{PROJECT_ROOT}/results"

# ============================================================
# TRAINING SETTINGS
# ============================================================
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

# ============================================================
# MODEL ARCHITECTURE SETTINGS
# ============================================================
VISUAL_DIM = 256
TEXT_DIM = 768  # PubMedBERT output dimension
HIDDEN_DIM = 256

# ============================================================
# TEXT ENCODER SETTINGS
# ============================================================
TEXT_ENCODER_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# ============================================================
# BUS DATASET - BI-RADS TEXT PROMPTS
# ============================================================
BIRADS_PROMPTS = {
    2: "Benign lesion with circumscribed smooth margins.",
    3: "Likely benign lesion with well-defined margins.",
    4: "Suspicious lesion with irregular margins or heterogeneous echogenicity.",
    5: "Highly suspicious malignant lesion with irregular shape, ill-defined margins, and posterior shadowing."
}

# ============================================================
# NSCLC DATASET - TNM STAGING TEXT PROMPTS
# ============================================================
T_STAGE_PROMPTS = {
    'Tis': "Segment a carcinoma in situ, very early stage with smooth margins.",
    'T1a': "Segment a small lung tumor ≤1cm, well-circumscribed with smooth margins.",
    'T1b': "Segment a lung tumor 1-2cm, generally well-defined borders.",
    'T1c': "Segment a lung tumor 2-3cm, may show slightly irregular margins.",
    'T2a': "Segment a lung tumor 3-4cm, moderate boundary irregularity.",
    'T2b': "Segment a lung tumor 4-5cm, irregular margins with possible satellite nodules.",
    'T3': "Segment a large tumor >5cm, highly irregular boundaries and possible invasion.",
    'T4': "Segment an advanced tumor with mediastinal invasion, complex irregular margins."
}

HISTOLOGY_PROMPTS = {
    'Adenocarcinoma': "This adenocarcinoma appears as ground-glass or part-solid nodule with spiculation.",
    'Squamous cell carcinoma': "This squamous cell carcinoma is centrally located with possible cavitation.",
    'NSCLC NOS (not otherwise specified)': "Non-small cell lung carcinoma, general characteristics.",
    'Large cell carcinoma': "Large cell carcinoma with prominent nucleoli."
}


# ============================================================
# BUS COUNTERFACTUAL EXPERIMENT SETTINGS
# ============================================================
BIRADS_COUNTERFACTUAL = {
    2: 5,  # Benign -> Highly suspicious 
    3: 5,  # Likely benign -> Highly suspicious
    4: 2, 
    5: 2, 
}

# ============================================================
# COUNTERFACTUAL EXPERIMENT SETTINGS
# ============================================================
T_STAGE_COUNTERFACTUAL = {
    'T1a': 'T4',   # smallest to largest
    'T1b': 'T4',
    'T1c': 'T3',
    'T2a': 'T4',
    'T2b': 'T1a',
    'T3': 'T1a',   # large to smallest
    'T4': 'T1a',   # largest to smallest
    'Tis': 'T4',
}

HISTOLOGY_COUNTERFACTUAL = {
    'Adenocarcinoma': 'Squamous cell carcinoma',
    'Squamous cell carcinoma': 'Adenocarcinoma',
    'NSCLC NOS (not otherwise specified)': 'Adenocarcinoma',
    'Large cell carcinoma': 'Adenocarcinoma'
}

# ============================================================
# HELPER FUNCTIONS - Get save paths
# ============================================================
def get_model_save_path(dataset_name="bus"):

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return f"{RESULTS_DIR}/{MODEL_NAME}_{dataset_name}_best.pth"

def get_results_save_path(experiment_type, dataset_name="bus"):

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return f"{RESULTS_DIR}/{MODEL_NAME}_{dataset_name}_{experiment_type}_results.csv"