import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# this function prepares a batch of images and captions to feed into the model
def collate(batch, vocabulary):
    images, captions = [], []

    for image, caption in batch:
        images.append(image)
        captions.append(caption)

    # stack images into a single tensor
    images = torch.stack(images, dim=0)
    # fill the captions with <pad> and make them same length
    captions = pad_sequence(captions, batch_first=True, padding_value=vocabulary.word2idx[vocabulary.PAD_TOKEN])
    return images, captions


def predict(model, image, vocabulary, device, maxLength=50, beamSize=5):
    model.eval()
    
    # get start and end token ids from the vocabulary
    startToken = vocabulary.word2idx[vocabulary.START_TOKEN]
    endToken = vocabulary.word2idx[vocabulary.END_TOKEN]
    vocabSize = len(vocabulary)

    with torch.no_grad():
        # encode the image
        encoderOut = model.encoder(image.unsqueeze(0).to(device))
        
        # Get feature dimensions from encoder
        encoderDim = encoderOut.size(-1)
        numPixels = encoderOut.size(1)
        encoderOut = encoderOut.view(1, numPixels, encoderDim)

        # copy encoder output beamSize times so each beam gets its own copy
        encoderOut = encoderOut.expand(beamSize, numPixels, encoderDim)

        # Number of active beams (candidates)
        k = beamSize
        
        # start all sequences with the <start> token
        seqs = torch.full((k, 1), startToken, dtype=torch.long, device=device)
        
        # keep running scores for each beam
        topKScores = torch.zeros(k, device=device)
        
        # store attention weights (alphas) for each sequence
        seqsAlpha = [torch.empty((0, numPixels), device=device) for _ in range(k)]

        # lists to save completed sequences and their scores
        completeSeqs = []
        completeScores = []
        completeSeqsAlpha = []

        # initialize the decoder's hidden and cell states
        h, c = model.decoder.initHiddenState(encoderOut)
        
        # keep track of how many decoding steps we've done
        step = 1
        # keep generating words until all sequences finish
        while True:
            # get embeddings of the last generated words for each sequence
            embeddings = model.decoder.embedding(seqs[:, -1])
            # apply attention mechanism to focus on image regions
            attentionWeightedEncoding, alpha = model.decoder.attention(encoderOut, h)
            # apply a gating scalar to control how much attention info is used
            gate = model.decoder.sigmoid(model.decoder.fBeta(h))
            attentionWeightedEncoding = gate * attentionWeightedEncoding

            # one decoding step: combine embeddings + attention + hidden states
            h, c = model.decoder.decodeStep(torch.cat([embeddings, attentionWeightedEncoding], dim=1), (h, c))
            # compute output scores
            scores = model.decoder.fc(model.decoder.dropoutLayer(h))
            # turn scores into log probabilities for easier math
            scores = F.log_softmax(scores, dim=1)

            # first step: only one sequence, so just pick top k words
            if step == 1:
                topKScores, topKWords = scores[0].topk(k, dim=0, largest=True)
                prevWordInds = torch.zeros(k, dtype=torch.long, device=device)
            else:
                # add previous scores to current log probs
                scores = scores + topKScores.unsqueeze(1)
                # flatten and pick the best k new sequences overall
                topKScores, topKWords = scores.view(-1).topk(k, dim=0, largest=True)
                # figure out which old sequence each new one came from
                prevWordInds = topKWords // vocabSize

            # find the actual next word indices for each beam
            nextWordInds = topKWords % vocabSize
            # append next words to their corresponding previous sequences
            seqs = torch.cat([seqs[prevWordInds], nextWordInds.unsqueeze(1)], dim=1)

            # update stored attention maps for each sequence
            newSeqsAlpha = []
            for idx, prevInd in enumerate(prevWordInds.tolist()):
                alphaSeq = torch.cat([seqsAlpha[prevInd], alpha[prevInd].unsqueeze(0)], dim=0)
                newSeqsAlpha.append(alphaSeq)
            seqsAlpha = newSeqsAlpha

            # reorder encoder output and hidden states to match surviving beams
            encoderOut = encoderOut[prevWordInds]
            h = h[prevWordInds]
            c = c[prevWordInds]

            # split finished vs unfinished sequences
            completeInds = []
            incompleteInds = []
            for ind, nextWord in enumerate(nextWordInds.tolist()):
                # a sequence is complete if it hits <end> or max length
                if nextWord == endToken or seqs[ind].size(0) >= maxLength:
                    completeInds.append(ind)
                else:
                    incompleteInds.append(ind)

            # save completed sequences and their scores
            if len(completeInds) > 0:
                completeSeqs.extend(seqs[completeInds].tolist())
                completeScores.extend(topKScores[completeInds])
                completeSeqsAlpha.extend([seqsAlpha[ind] for ind in completeInds])

            # reduce beam size for remaining incomplete sequences
            k -= len(completeInds)
            # stop if all sequences are done or max steps reached
            if k == 0 or step >= maxLength:
                break

            # keep only the incomplete sequences for next round
            seqs = seqs[incompleteInds]
            seqsAlpha = [seqsAlpha[ind] for ind in incompleteInds]
            encoderOut = encoderOut[incompleteInds]
            h = h[incompleteInds]
            c = c[incompleteInds]
            topKScores = topKScores[incompleteInds]

            # update beam count and move to next step
            k = len(seqs)
            step += 1

        # if no sequences completed, use current sequences
        if len(completeSeqs) == 0:
            completeSeqs = seqs.tolist()
            completeScores = topKScores
            completeSeqsAlpha = seqsAlpha
        else:
            # stack all final scores into one tensor for argmax
            completeScores = torch.stack(completeScores)

        # pick the sequence with the highest final score
        if isinstance(completeScores, torch.Tensor):
            bestIdx = completeScores.argmax().item()
        else:
            bestIdx = np.argmax(completeScores)

        # get best sequence and its attention weights
        bestSeq = completeSeqs[bestIdx]
        bestAlpha = completeSeqsAlpha[bestIdx].detach().cpu().numpy()

    # convert token ids back to words until <end> token is reached
    tokens = []
    for idx in bestSeq[1:]:
        if idx == endToken:
            break
        tokens.append(vocabulary.idx2word[idx])
    return ' '.join(tokens), bestAlpha