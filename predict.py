import cv2
import torch
import json
import mediapipe as mp
from torch_geometric.data import Data
from src.model import ASLGCN
from torch_geometric.nn import global_mean_pool

# === Load Label Map ===
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Invert the label map: index (as int) -> label
inv_label_map = {v: k for k, v in label_map.items()}

# === Load Trained Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLGCN(in_channels=2, hidden_channels=64, out_channels=len(label_map)).to(device)
model.load_state_dict(torch.load("models/model.pth", map_location=device))
model.eval()

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# === Define Edge Index for Hand Graph ===
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

# === Open Webcam ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            label = handedness.classification[0].label  # "Left" or "Right"

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Extract keypoints
            landmarks = hand_landmarks.landmark
            keypoints = torch.tensor([[lm.x, lm.y] for lm in landmarks], dtype=torch.float).to(device)

            if keypoints.shape[0] == 21:
                # Prepare graph data
                data = Data(x=keypoints, edge_index=edge_index)
                data.batch = torch.zeros(21, dtype=torch.long).to(device)

                # Run prediction
                with torch.no_grad():
                    out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1).item()
                    sign = inv_label_map.get(pred, "?")  # Use int key

                    # Debug
                    print(f"{label} hand -> predicted class {pred} -> {sign}")

                    # Display on frame
                    cv2.putText(frame, f"{label} Hand: {sign}", (10, 40 + idx * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hands detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("ASL GCN Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
