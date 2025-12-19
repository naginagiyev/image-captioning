import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ICDataset(Dataset):    
    def __init__(self, imagesDir, captionsDir, vocabulary, transform=None):
        self.imagesDir = imagesDir
        self.captionsDir = captionsDir
        self.vocabulary = vocabulary
        self.transform = transform
        # list to store image, caption pairs
        self.pairs = []

        # get all images
        imageFiles = [file for file in os.listdir(imagesDir) if file.endswith(('.jpg'))]
        
        for imgName in imageFiles:
            captionFile = os.path.splitext(imgName)[0] + '.txt'
            captionPath = os.path.join(self.captionsDir, captionFile)
            
            # read captions from text file
            with open(captionPath, 'r', encoding='utf-8') as f:
                captions = f.read().strip().split('\n')
            
            # add image, caption pairs to the list
            for caption in captions:
                if caption.strip():
                    self.pairs.append((imgName, caption))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        imgName, caption = self.pairs[idx]
        imgPath = os.path.join(self.imagesDir, imgName)
        image = Image.open(imgPath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # convert caption text into a list of word ids
        numericalizedCaption = [self.vocabulary.word2idx[self.vocabulary.START_TOKEN]]
        numericalizedCaption.extend(self.vocabulary.numericalize(caption))
        numericalizedCaption.append(self.vocabulary.word2idx[self.vocabulary.END_TOKEN])

        return image, torch.tensor(numericalizedCaption, dtype=torch.long)


# train augumentations
trainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # mean and standard deviation of each color channel (RGB) in the ImageNet dataset
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# evaluation augumentations
evalTransform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])