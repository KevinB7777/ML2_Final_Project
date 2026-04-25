import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


# --------------------------------------------------
# Device setup
# --------------------------------------------------
# Apple Silicon Macs use MPS, not CUDA.
# CUDA is for NVIDIA GPUs.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# Load pretrained ResNet18
# --------------------------------------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Remove the final classification layer.
# Original ResNet18:
# image -> convolution layers -> 512 features -> final 1000-class classifier
#
# We only want:
# image -> convolution layers -> 512 features
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Converts PIL image into PyTorch tensor:
    # H x W x C image -> C x H x W tensor
    # pixel values 0-255 -> 0-1
    transforms.ToTensor(),

    # ImageNet normalization.
    # ResNet18 was pretrained using this normalization,
    # so we use the same statistics.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


# --------------------------------------------------
# Output folder
# --------------------------------------------------
os.makedirs("data/features", exist_ok=True)


def extract_split(split):
    """
    Extract ResNet18 features for one split:
    train, val, or test.
    """

    input_path = f"data/processed/{split}_with_posters_tvdb.json"
    df = pd.read_json(input_path)

    features = []
    valid_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        poster_path = row["poster_path"]

        try:
            # Open image and force RGB format.
            # Some images might be grayscale or RGBA.
            img = Image.open(poster_path).convert("RGB")

            # Transform image into ResNet-ready tensor.
            # Shape after transform:
            # [3, 224, 224]
            x = transform(img)

            # Add batch dimension.
            # ResNet expects:
            # [batch_size, channels, height, width]
            #
            # Before unsqueeze:
            # [3, 224, 224]
            #
            # After unsqueeze(0):
            # [1, 3, 224, 224]
            x = x.unsqueeze(0)

            # Move image tensor to same device as model.
            x = x.to(device)

            # We are only extracting features, not training ResNet.
            # no_grad saves memory and speeds things up.
            with torch.no_grad():
                feat = feature_extractor(x)

            # ResNet output shape:
            # [1, 512, 1, 1]
            #
            # squeeze converts it to:
            # [512]
            feat = feat.squeeze().detach().cpu().numpy()

            features.append(feat)
            valid_rows.append(row)

        except Exception as e:
            print("Bad image:", poster_path, e)

    X = np.array(features)
    valid_df = pd.DataFrame(valid_rows)

    feature_path = f"data/features/X_image_{split}.npy"
    rows_path = f"data/features/{split}_image_rows.json"

    np.save(feature_path, X)
    valid_df.to_json(rows_path, orient="records", indent=2)

    print(split, "feature shape:", X.shape)
    print("saved:", feature_path)
    print("saved:", rows_path)


for split in ["train", "val", "test"]:
    extract_split(split)