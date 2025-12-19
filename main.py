import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import collate
from vocabulary import Vocabulary
from models import ImageCaptionModel
from preprocessing import split_data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import ICDataset, trainTransform, evalTransform
from trainer import trainEpoch, validate, calculateBLEUScores
from config import device, trainImages, trainCaptions, valImages, valCaptions, testImages, testCaptions, modelsDir

def main():
    # Split data into train/test/validation
    split_data()

    # Build vocabulary
    print("\n📚 Building vocabulary...")
    allCaptions = []
    for captionFile in tqdm(os.listdir(trainCaptions), desc="Reading captions"):
        captionPath = os.path.join(trainCaptions, captionFile)
        with open(captionPath, 'r', encoding='utf-8') as f:
            captions = f.read().strip().split('\n')
            allCaptions.extend(captions)

    vocabulary = Vocabulary(freqThresh=2)
    vocabulary.buildVocabulary(allCaptions)
    print(f"Vocabulary was created with size: {len(vocabulary)}")

    # Create datasets
    print("\n📦 Creating datasets...")
    trainDataset = ICDataset(trainImages, trainCaptions, vocabulary, transform=trainTransform)
    valDataset = ICDataset(valImages, valCaptions, vocabulary, transform=evalTransform)
    testDataset = ICDataset(testImages, testCaptions, vocabulary, transform=evalTransform)

    print(f"Train dataset size: {len(trainDataset)}")
    print(f"Val dataset size: {len(valDataset)}")
    print(f"Test dataset size: {len(testDataset)}")

    # Create DataLoaders
    print("\n🔄 Creating DataLoaders...")
    batchSize = 48
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, collate_fn=lambda batch: collate(batch, vocabulary), num_workers=num_workers)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, collate_fn=lambda batch: collate(batch, vocabulary), num_workers=num_workers)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, collate_fn=lambda batch: collate(batch, vocabulary), num_workers=num_workers)

    print(f"Train batches: {len(trainLoader)}")
    print(f"Validation batches: {len(valLoader)}")
    print(f"Test batches: {len(testLoader)}")

    # Initialize model
    print("\n🤖 Initializing model...")
    vocabSize = len(vocabulary)

    # define model dimensions
    attentionDim = 256
    embedDim = 256
    decoderDim = 512
    encoderDim = 256     

    # initalize full image captioning model
    model = ImageCaptionModel(
        attentionDim=attentionDim,
        embedDim=embedDim,
        decoderDim=decoderDim,
        vocabSize=vocabSize,
        encoderDim=encoderDim,
        dropout=0.50
    ).to(device)

    # Calculate number of parameters in ResNet layer4 that are being fine-tuned
    resnetFinetuned = sum(p.numel() for p in model.encoder.resnet[-1].parameters() if p.requires_grad)
    # Calculate parameters in the encoder's final linear layer
    encoderLinear = sum(p.numel() for p in model.encoder.linear.parameters())
    # Calculate total number of parameters in the decoder
    decoderParams = sum(p.numel() for p in model.decoder.parameters())

    # Print breakdown of where trainable parameters come from
    print(f"ResNet fine-tuned: {resnetFinetuned:,} parameters")
    print(f"Encoder linear layer: {encoderLinear:,} parameters")
    print(f"Decoder: {decoderParams:,} parameters")

    # Training setup
    print("\n⚙️ Setting up training...")
    # use crossentropy loss and ignore padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word2idx[vocabulary.PAD_TOKEN])

    # get trainable parameters
    encoderFinetunedParams = list(model.encoder.resnet[-1].parameters())
    encoderLinearParams = list(model.encoder.linear.parameters())
    encoderTrainableParams = encoderFinetunedParams + encoderLinearParams
    decoderParams = list(model.decoder.parameters())

    # create optimizer with different learning rates for encoder and decoder
    optimizer = torch.optim.Adam([
        {"params": decoderParams, "lr": 1e-4, "weight_decay": 3e-4},
        {"params": encoderTrainableParams, "lr": 1e-5, "weight_decay": 3e-4}
    ])

    print("Training setup is ready to run!")

    # Training loop
    print("\n🚀 Starting training...")
    numEpochs = 50
    bestValLoss = float('inf')

    patience = 10       
    patienceCounter = 0

    trainLosses = []
    valLosses = []

    bleu1Values = []
    bleu2Values = []
    bleu3Values = []
    bleu4Values = []

    scheduler = CosineAnnealingLR(optimizer, T_max=numEpochs, eta_min=1e-6)

    for epoch in range(numEpochs):
        print(f"\nEpoch {epoch+1}/{numEpochs}")
        
        # decrease teacher forcing gradually
        teacherForcingRatio = max(0.5, 1.0 - epoch * 0.02)

        # train for one epoch
        trainLoss = trainEpoch(model, trainLoader, criterion, optimizer, device, teacherForcingRatio, vocabulary)
        trainLosses.append(trainLoss)
        
        # validate after the epoch
        valLoss = validate(model, valLoader, criterion, device, vocabulary)
        valLosses.append(valLoss)

        # calculate bleu losses and append to corresponding lists 
        bleu1, bleu2, bleu3, bleu4 = calculateBLEUScores(model, valImages, valCaptions, vocabulary, device)
        bleu1Values.append(bleu1)
        bleu2Values.append(bleu2)
        bleu3Values.append(bleu3)
        bleu4Values.append(bleu4)
        
        # update learning rate
        scheduler.step()
        
        print(f"Train Loss: {trainLoss:.4f}, Val Loss: {valLoss:.4f}")
        print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, BLEU-3: {bleu3:.4f}, BLEU-4: {bleu4:.4f}")
        
        # save model if validation loss improved
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            patienceCounter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': valLoss,
                'vocab': vocabulary,
            }, os.path.join(modelsDir, 'bestmodel.pth'))
        else:
            patienceCounter += 1
            if patienceCounter >= patience:
                break

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()