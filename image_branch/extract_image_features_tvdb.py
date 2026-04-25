# import os
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import pandas as pd
# import numpy as np
# from tqdm import tqdm


# # --------------------------------------------------
# # Device setup
# # --------------------------------------------------
# # Apple Silicon Macs use MPS, not CUDA.
# # CUDA is for NVIDIA GPUs.
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


# # --------------------------------------------------
# # Load pretrained ResNet18
# # --------------------------------------------------
# resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# # Remove the final classification layer.
# # Original ResNet18:
# # image -> convolution layers -> 512 features -> final 1000-class classifier
# #
# # We only want:
# # image -> convolution layers -> 512 features
# feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# feature_extractor = feature_extractor.to(device)
# feature_extractor.eval()


# # --------------------------------------------------
# # Image preprocessing
# # --------------------------------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),

#     # Converts PIL image into PyTorch tensor:
#     # H x W x C image -> C x H x W tensor
#     # pixel values 0-255 -> 0-1
#     transforms.ToTensor(),

#     # ImageNet normalization.
#     # ResNet18 was pretrained using this normalization,
#     # so we use the same statistics.
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     )
# ])


# # --------------------------------------------------
# # Output folder
# # --------------------------------------------------
# os.makedirs("data/features", exist_ok=True)


# def extract_split(split):
#     """
#     Extract ResNet18 features for one split:
#     train, val, or test.
#     """

#     input_path = f"data/processed/{split}_with_posters_tvdb.json"
#     df = pd.read_json(input_path)

#     features = []
#     valid_rows = []

#     for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
#         poster_path = row["poster_path"]

#         try:
#             # Open image and force RGB format.
#             # Some images might be grayscale or RGBA.
#             img = Image.open(poster_path).convert("RGB")

#             # Transform image into ResNet-ready tensor.
#             # Shape after transform:
#             # [3, 224, 224]
#             x = transform(img)

#             # Add batch dimension.
#             # ResNet expects:
#             # [batch_size, channels, height, width]
#             #
#             # Before unsqueeze:
#             # [3, 224, 224]
#             #
#             # After unsqueeze(0):
#             # [1, 3, 224, 224]
#             x = x.unsqueeze(0)

#             # Move image tensor to same device as model.
#             x = x.to(device)

#             # We are only extracting features, not training ResNet.
#             # no_grad saves memory and speeds things up.
#             with torch.no_grad():
#                 feat = feature_extractor(x)

#             # ResNet output shape:
#             # [1, 512, 1, 1]
#             #
#             # squeeze converts it to:
#             # [512]
#             feat = feat.squeeze().detach().cpu().numpy()

#             features.append(feat)
#             valid_rows.append(row)

#         except Exception as e:
#             print("Bad image:", poster_path, e)

#     X = np.array(features)
#     valid_df = pd.DataFrame(valid_rows)

#     feature_path = f"data/features/X_image_{split}.npy"
#     rows_path = f"data/features/{split}_image_rows.json"

#     np.save(feature_path, X)
#     valid_df.to_json(rows_path, orient="records", indent=2)

#     print(split, "feature shape:", X.shape)
#     print("saved:", feature_path)
#     print("saved:", rows_path)


# for split in ["train", "val", "test"]:
#     extract_split(split)

import os
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

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


# Resolve paths relative to this script so execution does not depend on cwd.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"

# Optional smoke-test limiter: set MAX_ROWS to process only first N rows/split.
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))


# --------------------------------------------------
# Load pretrained ResNet18
# --------------------------------------------------
print("Loading ResNet18 model...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Remove the final classification layer to get the 512 feature vector
resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet_feature_extractor = resnet_feature_extractor.to(device)
resnet_feature_extractor.eval()

# ResNet Image preprocessing (ImageNet standards)
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

# --------------------------------------------------
# Load pretrained CLIP (Vision-Language Semantic Features)
# --------------------------------------------------
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def _clip_output_to_numpy(clip_output):
    """
    Convert CLIP output to a 1D numpy embedding.

    Depending on transformers version, get_image_features may return
    either a Tensor directly or a model output object with pooler_output.
    """
    if isinstance(clip_output, torch.Tensor):
        vec = clip_output
    elif hasattr(clip_output, "pooler_output") and clip_output.pooler_output is not None:
        vec = clip_output.pooler_output
    elif hasattr(clip_output, "last_hidden_state") and clip_output.last_hidden_state is not None:
        # Fallback mean-pooling if pooler_output is unavailable.
        vec = clip_output.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unexpected CLIP output type: {type(clip_output)}")

    return vec.squeeze(0).detach().cpu().numpy()


# --------------------------------------------------
# Output folder
# --------------------------------------------------
os.makedirs(FEATURES_DIR, exist_ok=True)


def extract_split(split):
    """
    Extract ResNet18 (structural) + CLIP (semantic) features for one split:
    train, val, or test.
    """

    input_path = PROCESSED_DIR / f"{split}_with_posters_tvdb.json"
    df = pd.read_json(input_path)
    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS)

    features = []
    valid_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
        poster_path = Path(row["poster_path"])
        if not poster_path.is_absolute():
            poster_path = BASE_DIR / poster_path

        try:
            # Open image and force RGB format
            img = Image.open(poster_path).convert("RGB")

            # -----------------------------------------
            # 1. Extract ResNet18 features
            # -----------------------------------------
            x_resnet = resnet_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                resnet_feat = resnet_feature_extractor(x_resnet)
            
            # Squeeze to shape (512,)
            resnet_feat = resnet_feat.squeeze().detach().cpu().numpy()

            # -----------------------------------------
            # 2. Extract CLIP features
            # -----------------------------------------
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                clip_out = clip_model.get_image_features(**inputs)
            
            # Convert CLIP output object/tensor to shape (512,)
            clip_feat = _clip_output_to_numpy(clip_out)
            
            # Normalize CLIP embeddings (standard practice for CLIP models)
            clip_norm = np.linalg.norm(clip_feat)
            if clip_norm > 0:
                clip_feat = clip_feat / clip_norm

            # -----------------------------------------
            # 3. Combine Features (Late Fusion Preparation)
            # -----------------------------------------
            # Final feature vector: 512 (ResNet) + 512 (CLIP) = 1024 dimensions
            combined_feat = np.concatenate([resnet_feat, clip_feat])

            features.append(combined_feat)
            valid_rows.append(row)

        except Exception as e:
            print(f"Bad image: {poster_path} - Error: {e}")

    X = np.array(features)
    valid_df = pd.DataFrame(valid_rows)

    feature_path = FEATURES_DIR / f"X_image_{split}.npy"
    rows_path = FEATURES_DIR / f"{split}_image_rows.json"

    # Save the new 1024-dimensional arrays
    np.save(feature_path, X)
    valid_df.to_json(rows_path, orient="records", indent=2)

    print(f"{split} feature shape: {X.shape}") # You should expect to see (N, 1024) here
    print(f"saved: {feature_path}")
    print(f"saved: {rows_path}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        extract_split(split)