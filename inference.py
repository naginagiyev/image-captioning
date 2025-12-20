import os
import torch
import random
from PIL import Image
from utils import predict
from dataset import evalTransform
from vocabulary import Vocabulary
from models import ImageCaptionModel
from config import device, modelsDir, testImages

def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(modelsDir, 'bestmodel.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    vocabulary = checkpoint['vocab']
    vocabSize = len(vocabulary)
    
    attentionDim = 256
    embedDim = 256
    decoderDim = 512
    encoderDim = 256
    
    model = ImageCaptionModel(
        attentionDim=attentionDim,
        embedDim=embedDim,
        decoderDim=decoderDim,
        vocabSize=vocabSize,
        encoderDim=encoderDim,
        dropout=0.50
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocabulary


def generate_caption(image_path=None, model=None, vocabulary=None, checkpoint_path=None, beam_size=5, max_length=50):
    if model is None or vocabulary is None:
        model, vocabulary = load_model(checkpoint_path)
    
    if image_path is None:
        image_files = [f for f in os.listdir(testImages) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            raise FileNotFoundError(f"No images found in {testImages}")
        image_path = os.path.join(testImages, random.choice(image_files))
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = evalTransform(image)
    
    caption, attention_weights = predict(
        model, 
        image_tensor, 
        vocabulary, 
        device, 
        maxLength=max_length, 
        beamSize=beam_size
    )
    
    return image_path, caption


if __name__ == "__main__":
    image_path = input("Enter image path (or press Enter for random): ").strip()
    if not image_path:
        image_path = None
    
    image_path, caption = generate_caption(image_path)
    
    print(f"🖼️  Image: {os.path.basename(image_path)}")
    print(f"📝 Caption: {caption}")