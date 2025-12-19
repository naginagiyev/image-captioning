import os
import torch
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from dataset import evalTransform
from sacrebleu.metrics import BLEU
from torch.nn.utils.rnn import pack_padded_sequence

# function to train model for one epoch
def trainEpoch(model, dataLoader, criterion, optimizer, device, teacherForcingRatio, vocabulary):
    model.train()
    totalLoss = 0

    # loop over the batches of images
    for images, captions in tqdm(dataLoader, desc="Training"):
        images = images.to(device)
        captions = captions.to(device)

        # compute how many real tokens are in the caption (without padding tokens)
        captionLengths = torch.tensor([torch.sum(cap != vocabulary.word2idx[vocabulary.PAD_TOKEN]).item() for cap in captions]).unsqueeze(1).to(device)
        # forward pass
        predictions, encodedCaptions, decodeLengths, sortInd, alphas = model(images, captions, captionLengths, teacherForcingRatio=teacherForcingRatio)
        targets = encodedCaptions[:, 1:]

        # pack sequences to remove padding before computing loss
        predictions = pack_padded_sequence(predictions, decodeLengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data
        # compute loss between predicted and target words
        loss = criterion(predictions, targets)

        # clear old gradients
        optimizer.zero_grad()
        # backpropagate to compute gradients
        loss.backward()
        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        # update weights using optimizer
        optimizer.step()
        totalLoss += loss.item()

    return totalLoss / len(dataLoader)


# function to evaluate model
def validate(model, dataLoader, criterion, device, vocabulary):
    model.eval()
    totalLoss = 0

    # disable gradient calculations for faster validation
    with torch.no_grad():
        for images, captions in tqdm(dataLoader, desc="Computing Loss"):
            images = images.to(device)
            captions = captions.to(device)

            # compute caption lengths
            captionLengths = torch.tensor([torch.sum(cap != vocabulary.word2idx[vocabulary.PAD_TOKEN]).item() for cap in captions]).unsqueeze(1).to(device)
            # Forward pass through encoder and decoder
            predictions, encodedCaptions, decodeLengths, sortInd, alphas = model(images, captions, captionLengths, teacherForcingRatio=1.0)
            targets = encodedCaptions[:, 1:]

            # pack sequences
            predictions = pack_padded_sequence(predictions, decodeLengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data

            # compute validation loss
            loss = criterion(predictions, targets)
            totalLoss += loss.item()

    return totalLoss / len(dataLoader)


def calculateBLEUScores(model, valImagesDir, valCaptionsDir, vocabulary, device):
    model.eval()
    bleu1Metric = BLEU(max_ngram_order=1)
    bleu2Metric = BLEU(max_ngram_order=2)
    bleu3Metric = BLEU(max_ngram_order=3)
    bleu4Metric = BLEU(max_ngram_order=4)
    
    allPredictions = []
    allReferences = []
    
    imageFiles = [f for f in os.listdir(valImagesDir) if f.endswith('.jpg')]
    
    with torch.no_grad():
        for imgName in tqdm(imageFiles, desc="Computing BLEU"):
            imgPath = os.path.join(valImagesDir, imgName)
            capFile = os.path.splitext(imgName)[0] + '.txt'
            capPath = os.path.join(valCaptionsDir, capFile)
            
            image = Image.open(imgPath).convert('RGB')
            imageTensor = evalTransform(image).unsqueeze(0).to(device)
            
            with open(capPath, 'r', encoding='utf-8') as f:
                trueCaptions = [line.strip() for line in f.readlines() if line.strip()]
            
            encoderOut = model.encoder(imageTensor)
            encoderDim = encoderOut.size(-1)
            numPixels = encoderOut.size(1)
            encoderOut = encoderOut.view(1, numPixels, encoderDim)
            
            startToken = vocabulary.word2idx[vocabulary.START_TOKEN]
            endToken = vocabulary.word2idx[vocabulary.END_TOKEN]
            
            h, c = model.decoder.initHiddenState(encoderOut)
            seq = [startToken]
            
            for _ in range(50):
                embeddings = model.decoder.embedding(torch.tensor([seq[-1]], device=device))
                attentionWeightedEncoding, alpha = model.decoder.attention(encoderOut, h)
                gate = model.decoder.sigmoid(model.decoder.fBeta(h))
                attentionWeightedEncoding = gate * attentionWeightedEncoding
                h, c = model.decoder.decodeStep(torch.cat([embeddings, attentionWeightedEncoding], dim=1), (h, c))
                scores = model.decoder.fc(h)
                nextWord = scores.argmax(1).item()
                if nextWord == endToken:
                    break
                seq.append(nextWord)
            
            tokens = []
            for idx in seq[1:]:
                if idx == endToken:
                    break
                tokens.append(vocabulary.idx2word[idx])
            predCaption = ' '.join(tokens)
            
            allPredictions.append(predCaption)
            allReferences.append(trueCaptions)
    
    numRefs = max(len(refs) for refs in allReferences)
    transposedRefs = []
    for refIdx in range(numRefs):
        transposedRefs.append([refs[refIdx] if refIdx < len(refs) else refs[0] for refs in allReferences])
    
    bleu1Score = bleu1Metric.corpus_score(allPredictions, transposedRefs).score
    bleu2Score = bleu2Metric.corpus_score(allPredictions, transposedRefs).score
    bleu3Score = bleu3Metric.corpus_score(allPredictions, transposedRefs).score
    bleu4Score = bleu4Metric.corpus_score(allPredictions, transposedRefs).score
    
    return bleu1Score, bleu2Score, bleu3Score, bleu4Score

