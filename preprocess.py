import os
import cv2
import torch
import json
import mediapipe as mp
from tqdm import tqdm
from torch_geometric.data import Data

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/asl_alphabet_train"))  # or test folder
SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/graphs"))
LABEL_MAP_PATH = os.path.abspath(os.path.join(BASE_DIR, "../label_map.json"))

os.makedirs(SAVE_DIR, exist_ok=True)

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# === Define edges (MediaPipe hand skeleton) ===
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

def extract_keypoints(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return torch.tensor([[lm.x, lm.y] for lm in landmarks], dtype=torch.float)
    return None

def preprocess_and_save():
    # Get all class folders
    label_folders = [f for f in sorted(os.listdir(ROOT_DIR)) if os.path.isdir(os.path.join(ROOT_DIR, f))]
    label_map = {label: idx for idx, label in enumerate(label_folders)}

    print(f"Found {len(label_map)} classes.")
    print("Saving label_map.json...")
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)

    for label in tqdm(label_map, desc="Processing Classes"):
        class_dir = os.path.join(ROOT_DIR, label)
        for fname in os.listdir(class_dir):
            if not fname.endswith((".jpg", ".png")):
                continue

            path = os.path.join(class_dir, fname)
            image = cv2.imread(path)
            if image is None:
                print(f"Could not read image: {path}")
                continue

            keypoints = extract_keypoints(image)
            if keypoints is None or keypoints.shape[0] != 21:
                continue

            data = Data(x=keypoints, edge_index=edge_index, y=torch.tensor([label_map[label]]))
            save_path = os.path.join(SAVE_DIR, f"{label}_{fname.split('.')[0]}.pt")
            torch.save(data, save_path)

if __name__ == "__main__":
    print("Starting keypoint extraction and graph creation...")
    preprocess_and_save()
    print(f"Done. Graphs saved in: {SAVE_DIR}")
