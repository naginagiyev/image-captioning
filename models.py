import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# here we'll partially fine-tune resnet model...
class EncoderCNN(nn.Module):
    def __init__(self, embedSize=256):
        super(EncoderCNN, self).__init__()
        # load resnet model
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # remove the last two layers (avgpool and fully connected layer)
        # because we only need the convolutional part for feature extraction
        modules = list(resnet.children())[:-2]
        # combine the remaining layers in one single sequential module
        self.resnet = nn.Sequential(*modules)

        # freeze all layers 
        for param in self.resnet.parameters():
            param.requires_grad = False

        # unfreeze the last convolution block to fine-tune it
        for param in self.resnet[-1].parameters():  
            param.requires_grad = True

        # fix the size of feature maps 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        # linear layer to shrink vectors
        self.linear = nn.Linear(resnet.fc.in_features, embedSize)

    def forward(self, images):
        with torch.set_grad_enabled(self.training):
            # send image to resnet to get feature maps
            features = self.resnet(images)
        features = self.adaptive_pool(features)
        # change the order of dimensions
        features = features.permute(0, 2, 3, 1)
        batch_size = features.size(0)
        # flatten 14x14 grid into a vector
        features = features.view(batch_size, -1, features.size(-1))
        features = self.linear(features)
        return features


class Attention(nn.Module):
    def __init__(self, encoderDim, decoderDim, attentionDim):
        super(Attention, self).__init__()
        
        # convert encoder and decoder features into smaller attention features
        self.encoderAtt = nn.Linear(encoderDim, attentionDim)
        self.decoderAtt = nn.Linear(decoderDim, attentionDim)
        
        # combine encoder, decoder and make a single score for each image region
        self.fullAtt = nn.Linear(attentionDim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, encoderOut, decoderHidden):
        
        # get attention features from encoder and decoder output
        att1 = self.encoderAtt(encoderOut)
        att2 = self.decoderAtt(decoderHidden)
        
        # add them together, apply relu, and make a score for each pixel
        att = self.fullAtt(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        
        # apply dropout if it is training
        att = self.dropout(att) if self.training else att
        
        # turn scores into probabilities
        alpha = self.softmax(att)
        
        # multiply image features by their attention weights and sum up
        attentionWeightedEncoding = (encoderOut * alpha.unsqueeze(2)).sum(dim=1)
        return attentionWeightedEncoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, attentionDim, embedDim, decoderDim, vocabSize, encoderDim=256, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.attentionDim = attentionDim
        self.embedDim = embedDim
        self.decoderDim = decoderDim
        self.vocabSize = vocabSize
        self.encoderDim = encoderDim
        self.dropout = dropout

        # create the attention model
        self.attention = Attention(encoderDim, decoderDim, attentionDim)
        # turn ids into word embeddings
        self.embedding = nn.Embedding(vocabSize, embedDim)
        # drop out some neurons randomly to prevent overfitting
        self.embeddingDropout = nn.Dropout(p=self.dropout * 0.5)
        self.dropoutLayer = nn.Dropout(p=self.dropout)

        # lstm that predicts next word using current word + image info
        self.decodeStep = nn.LSTMCell(embedDim + encoderDim, decoderDim, bias=True)

        # initialize hidden state and cell state
        self.initH = nn.Linear(encoderDim, decoderDim)
        self.initC = nn.Linear(encoderDim, decoderDim)

        # decide how much image info to use at each step
        self.fBeta = nn.Linear(decoderDim, encoderDim)
        self.sigmoid = nn.Sigmoid()

        # final layer that predicts the next word
        self.fc = nn.Linear(decoderDim, vocabSize)
        self.initWeights()

    def initWeights(self):
        # set small random values for embedding weights
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # set all final layer biases to zero
        self.fc.bias.data.fill_(0)
        # set small random values for final layer weights
        self.fc.weight.data.uniform_(-0.1, 0.1)

    # using encoder output initalize the hidden state and cell state
    def initHiddenState(self, encoderOut):
        batchSize = encoderOut.size(0)
        attentionWeightedEncoding, _ = self.attention(encoderOut, torch.zeros(batchSize, self.decoderDim, device=encoderOut.device))
        h = self.initH(attentionWeightedEncoding)
        c = self.initC(attentionWeightedEncoding)
        return h, c

    def forward(self, encoderOut, encodedCaptions, captionLengths, teacherForcingRatio=1.0):
        # get batch size and feature dimensions
        batchSize = encoderOut.size(0)
        encoderDim = encoderOut.size(-1)
        vocabSize = self.vocabSize

        # flatten image features
        encoderOut = encoderOut.view(batchSize, -1, encoderDim)
        numPixels = encoderOut.size(1)

        # sort captions by length
        captionLengths, sortInd = captionLengths.squeeze(1).sort(dim=0, descending=True)

        # reorder encoder outputs and captions by sorted indices
        encoderOut = encoderOut[sortInd]
        encodedCaptions = encodedCaptions[sortInd]

        # turn words into embeddings
        embeddings = self.embedding(encodedCaptions)

        # apply dropout to embeddings during training
        embeddings = self.embeddingDropout(embeddings) if self.training else embeddings

        # get initial hidden and cell states
        h, c = self.initHiddenState(encoderOut)

        # compute how long each caption (minus <end> token)
        decodeLengths = (captionLengths - 1).tolist()

        # prepare empty tensors to store predictions and attention weights
        predictions = torch.zeros(batchSize, max(decodeLengths), vocabSize, device=encoderOut.device)
        alphas = torch.zeros(batchSize, max(decodeLengths), numPixels, device=encoderOut.device)

        # start decoding with <start> tokens
        prevWords = encodedCaptions[:, 0].clone()

        # loop through each time step
        for t in range(max(decodeLengths)):
            # only keep samples that still have words left
            batchSizeT = sum([l > t for l in decodeLengths])

            # apply attention to focus on image parts for current step
            attentionWeightedEncoding, alpha = self.attention(encoderOut[:batchSizeT], h[:batchSizeT])

            # gate to control how much of the image info we use
            gate = self.sigmoid(self.fBeta(h[:batchSizeT]))
            attentionWeightedEncoding = gate * attentionWeightedEncoding

            # choose input word embedding for this step
            if t == 0:
                # first word always uses <start> token
                currentEmbeddings = embeddings[:batchSizeT, t, :]
            else:
                # randomly decide to use teacher forcing (true word) or model's prediction
                teacherMask = torch.rand(batchSizeT, device=encoderOut.device) < teacherForcingRatio

                # create empty space for embeddings
                currentEmbeddings = torch.zeros(batchSizeT, self.embedDim, device=encoderOut.device)

                # use real word embeddings when teacher forcing applies
                if teacherMask.any():
                    currentEmbeddings[teacherMask] = embeddings[:batchSizeT, t, :][teacherMask]

                # use model's own predicted words otherwise
                if (~teacherMask).any():
                    sampledTokens = prevWords[:batchSizeT][~teacherMask]
                    emb = self.embedding(sampledTokens)
                    if self.training:
                        emb = self.embeddingDropout(emb)
                    currentEmbeddings[~teacherMask] = emb

            # run one decoding step through LSTMCell
            h, c = self.decodeStep(
                torch.cat([currentEmbeddings, attentionWeightedEncoding], dim=1),
                (h[:batchSizeT], c[:batchSizeT])
            )

            # predict next word from lstm output
            preds = self.fc(self.dropoutLayer(h))

            # save predictions and attention weights
            predictions[:batchSizeT, t, :] = preds
            alphas[:batchSizeT, t, :] = alpha

            # update previous words with newly predicted tokens
            newPrevWords = prevWords.clone()
            newPrevWords[:batchSizeT] = preds.argmax(1)
            prevWords = newPrevWords

        # return predictions, captions, decode lengths, sorting info, and attention maps
        return predictions, encodedCaptions, decodeLengths, sortInd, alphas


# this model combines our encoder, decoder and attention modules in one pipeline 
class ImageCaptionModel(nn.Module):
    def __init__(self, attentionDim=256, embedDim=256, decoderDim=512, vocabSize=None, encoderDim=256, dropout=0.5):
        super(ImageCaptionModel, self).__init__()
        
        # create the cnn encoder based on resnet and...
        # rnn decoder with attention mechanism
        self.encoder = EncoderCNN(encoderDim)
        self.decoder = DecoderRNN(attentionDim, embedDim, decoderDim, vocabSize, encoderDim, dropout)

    def forward(self, images, encodedCaptions, captionLengths, teacherForcingRatio=1.0):
        # pass image to encoder cnn to get feature maps
        encoderOut = self.encoder(images)
        
        # pass encoder outputs and captions to decoder to predict next words
        predictions, encodedCaptions, decodeLengths, sortInd, alphas = self.decoder(
            encoderOut, encodedCaptions, captionLengths,
            teacherForcingRatio=teacherForcingRatio
        )

        # return decoder's predictions
        return predictions, encodedCaptions, decodeLengths, sortInd, alphas