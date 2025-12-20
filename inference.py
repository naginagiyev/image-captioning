import os
import torch
from PIL import Image
from models import ImageCaptionModel
from utils import predict
from dataset import evalTransform
from config import device, modelsDir


def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(modelsDir, 'bestmodel.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Please train the model first.")
    
    print(f"📦 Loading model from {checkpoint_path}...")
    
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
    
    print(f"✅ Model loaded successfully!")
    print(f"   Vocabulary size: {vocabSize}")
    print(f"   Device: {device}")
    
    return model, vocabulary


def generate_caption(image_path, model=None, vocabulary=None, checkpoint_path=None, beam_size=5, max_length=50):
    if model is None or vocabulary is None:
        model, vocabulary = load_model(checkpoint_path)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    print(f"\n🖼️  Processing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = evalTransform(image)
    
    print("🧠 Generating caption...")
    caption, attention_weights = predict(
        model, 
        image_tensor, 
        vocabulary, 
        device, 
        maxLength=max_length, 
        beamSize=beam_size
    )
    
    return caption


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    
    caption = generate_caption(image_path)
    
    print("\n" + "="*60)
    print("📝 Generated Caption:")
    print("="*60)
    print(caption)
    print("="*60 + "\n")

