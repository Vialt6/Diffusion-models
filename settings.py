import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
from torch import optim
import sys

SEED = 42
BATCH_SIZE = 128
N_EPOCHS = 150
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOISE_STEPS = 1000 
BETA_START = 1e-4
BETA_END = 0.02
IMAGE_CHW = (3, 64, 64)
NUM_CLASSES = 22
EMA_BETA = 0.995

MODEL_PATH = "./models"
RESULTS_PATH = "./results"
OUTPUT_PATH = "./output"
DATASET_PATH = "./dataset"
CLASS_LEGEND_FILE = "legenda_classi.json"
TRAIN_FILE = "train_full.txt"
TEST_FILE = "test.txt"
MODEL_FILE = "ckpt.pt"
EMA_MODEL_FILE = "ema_ckpt.pt"
OPTIM_FILE = "optim.pt"
NUM_EPOCH_FILE = "./num_epoch.txt"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)