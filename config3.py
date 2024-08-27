FRAC_TRAIN = 1
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
# STEP_SIZE = 40
MILESTONES = [15, 22]
GAMMA = 0.1
SEED = 30
NUM_CLASSES = 7
SAMPLE_DURATION = 28
MODEL_PATH = "resnet-34-kinetics-cpu.pth"
DATA_TYPE = "real_0.05"
SAVE_DIR = f"models/models_random_3_{DATA_TYPE}"
MODEL_NAME = f"resnet_{FRAC_TRAIN}.pth"
TRAIN_PATH = f"/local/scratch/c_xingjia2/sabhay/training_3d_resnet/finetuning/global_data/train_real_split_2_0.05.csv"
VAL_PATH = f"/local/scratch/c_xingjia2/sabhay/training_3d_resnet/finetuning/global_data/test_real_split_2.csv"
TEST_PATH = f"/local/scratch/c_xingjia2/sabhay/training_3d_resnet/finetuning/global_data/test_real_split_2.csv"
DATASET = "experimental_data"
MEAN = 0
STD = 1


# resnet-34-kinetics-cpu.pth

# Frac = 1 -> Epochs = 10, step_size = 2, lr = 1e-4
# Frac = 0.25 -> Epochs = 12, step_size = 3, lr = 1e-4
# Random init -> Epochs = 30, step_size = 5 or 6, lr = 1e-3

# SynapseMNIST3D
# MEAN = 0.5098059773445129
# STD = 0.21346217393875122

# OrganMNIST3D
# MEAN = 0.5003777742385864
# STD = 0.2760401964187622

# NoduleMNIST3D
# MEAN = 0.2686219811439514
# STD = 0.2722104787826538
