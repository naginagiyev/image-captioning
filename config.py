import os
import nltk
import torch
import warnings

warnings.filterwarnings("ignore")
nltk.data.find('tokenizers/punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Folder Paths
projectDir = os.getcwd()
dataDir = os.path.join(projectDir, "data")
imagesDir = os.path.join(dataDir, "images")
modelsDir = os.path.join(projectDir, "models")
trainImages = os.path.join(dataDir, "trainImages")
trainCaptions = os.path.join(dataDir, "trainCaptions")
testImages = os.path.join(dataDir, "testImages")
testCaptions = os.path.join(dataDir, "testCaptions")
valImages = os.path.join(dataDir, "valImages")
valCaptions = os.path.join(dataDir, "valCaptions")