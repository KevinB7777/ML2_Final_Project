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
if torch.backends.mps.is_available():
    compute_device = torch.device("mps")
elif torch.cuda.is_available():
    compute_device = torch.device("cuda")
else:
    compute_device = torch.device("cpu")

print("Using device:", compute_device)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))

print("Loading ResNet18 model...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet_feature_extractor = resnet_feature_extractor.to(compute_device)
resnet_feature_extractor.eval()

resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(compute_device)
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def clip_output_to_numpy_vector(clip_output):
    if isinstance(clip_output, torch.Tensor):
        vec = clip_output
    elif hasattr(clip_output, "pooler_output") and clip_output.pooler_output is not None:
        vec = clip_output.pooler_output
    elif hasattr(clip_output, "last_hidden_state") and clip_output.last_hidden_state is not None:
        vec = clip_output.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unexpected CLIP output type: {type(clip_output)}")

    return vec.squeeze(0).detach().cpu().numpy()


os.makedirs(FEATURES_DIR, exist_ok=True)


def extract_image_features_for_split(split_name):
    split_json_path = PROCESSED_DIR / f"{split_name}_with_posters_tvdb.json"
    split_df = pd.read_json(split_json_path)
    if MAX_ROWS > 0:
        split_df = split_df.head(MAX_ROWS)

    feature_rows = []
    kept_rows = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
        poster_path = Path(row["poster_path"])
        if not poster_path.is_absolute():
            poster_path = SCRIPT_DIR / poster_path

        try:
            img = Image.open(poster_path).convert("RGB")
            resnet_input_batch = resnet_transform(img).unsqueeze(0).to(compute_device)
            with torch.no_grad():
                resnet_feat = resnet_feature_extractor(resnet_input_batch)
            
            resnet_embedding = resnet_feat.squeeze().detach().cpu().numpy()
            clip_inputs = clip_processor(images=img, return_tensors="pt").to(compute_device)
            with torch.no_grad():
                clip_output = clip_model.get_image_features(**clip_inputs)
            clip_embedding = clip_output_to_numpy_vector(clip_output)
            clip_embedding_norm = np.linalg.norm(clip_embedding)
            if clip_embedding_norm > 0:
                clip_embedding = clip_embedding / clip_embedding_norm
            combined_embedding = np.concatenate([resnet_embedding, clip_embedding])

            feature_rows.append(combined_embedding)
            kept_rows.append(row)

        except Exception as e:
            print(f"Bad image: {poster_path} - Error: {e}")

    feature_matrix = np.array(feature_rows)
    kept_df = pd.DataFrame(kept_rows)

    feature_matrix_path = FEATURES_DIR / f"X_image_{split_name}.npy"
    kept_rows_path = FEATURES_DIR / f"{split_name}_image_rows.json"
    np.save(feature_matrix_path, feature_matrix)
    kept_df.to_json(kept_rows_path, orient="records", indent=2)

    print(f"{split_name} feature shape: {feature_matrix.shape}")
    print(f"saved: {feature_matrix_path}")
    print(f"saved: {kept_rows_path}")


if __name__ == "__main__":
    for split_name in ["train", "val", "test"]:
        extract_image_features_for_split(split_name)