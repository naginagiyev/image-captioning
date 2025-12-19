import os
import shutil
import pandas as pd
from tqdm import tqdm
from config import dataDir, imagesDir, trainImages, trainCaptions, testImages, testCaptions, valImages, valCaptions

def split_data():
    # create folders if not exist
    for folder in [trainImages, trainCaptions, testImages,
                   testCaptions, valImages, valCaptions]:
        os.makedirs(folder, exist_ok=True)

    # the number of samples for train/test/val
    trainSize, testSize, valSize = 6500, 500, 1091

    # read the captions data, group them and shuffle
    captionsDF = pd.read_csv(os.path.join(dataDir, "captions.txt"))
    grouped = captionsDF.groupby("image")["caption"].apply(list).reset_index()
    grouped = grouped.sample(frac=1, random_state=42).reset_index(drop=True)

    # split to sub dataframes based on train/test/val sizes
    train_data = grouped.iloc[:trainSize]
    test_data = grouped.iloc[trainSize:trainSize + testSize]
    val_data = grouped.iloc[trainSize + testSize:]

    # copy train images and captions 
    print("📦 Moving data...")
    for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Train split"):
        if os.path.exists(os.path.join(imagesDir, row["image"])):
            shutil.copy(os.path.join(imagesDir, row["image"]), os.path.join(trainImages, row["image"]))

        captionFile = os.path.splitext(row["image"])[0] + ".txt"
        with open(os.path.join(trainCaptions, captionFile), "w", encoding="utf-8") as f:
            for caption in row["caption"]:
                f.write(caption.strip() + "\n")

    # copy test images and captions 
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Test split"):
        if os.path.exists(os.path.join(imagesDir, row["image"])):
            shutil.copy(os.path.join(imagesDir, row["image"]), os.path.join(testImages, row["image"]))

        cap_file = os.path.splitext(row["image"])[0] + ".txt"
        with open(os.path.join(testCaptions, cap_file), "w", encoding="utf-8") as f:
            for caption in row["caption"]:
                f.write(caption.strip() + "\n")

    # copy validation images and captions 
    for _, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Val split"):
        if os.path.exists(os.path.join(imagesDir, row["image"])):
            shutil.copy(os.path.join(imagesDir, row["image"]), os.path.join(valImages, row["image"]))

        cap_file = os.path.splitext(row["image"])[0] + ".txt"
        with open(os.path.join(valCaptions, cap_file), "w", encoding="utf-8") as f:
            for caption in row["caption"]:
                f.write(caption.strip() + "\n")

    print("✅ Data successfully split into train/test/val folders!")