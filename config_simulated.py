FRAC_TRAIN = 1
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0
# STEP_SIZE = 40
MILESTONES = [15, 22]
GAMMA = 0.1
SEED = 30
NUM_CLASSES = 10
SAMPLE_DURATION = 32
MODEL_PATH = "" #resnet-34-kinetics-cpu.pth"
DATA_TYPE = "exp"
DATA_SNR = "005"
SAVE_DIR = f"models/models_rgb_{DATA_TYPE}"
MODEL_NAME = f"resnet_{DATA_SNR}_{FRAC_TRAIN}.pth"
TRAIN_PATH = f"./data_split/simulated_SNR{DATA_SNR}_train.csv"
VAL_PATH = f"./data_split/simulated_SNR{DATA_SNR}_val.csv"
TEST_PATH = f"./data_split/simulated_SNR{DATA_SNR}_test.csv"
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
