import cv2
import torch
import json
import mediapipe as mp
import numpy as np
from torchvision import transforms
from model2 import SimpleASLCNN

# === Load Label Map ===
with open("label_map.json", "r") as f:
    label_map = json.load(f)
# Invert the label map: index (as int) -> label
inv_label_map = {v: k for k, v in label_map.items()}

# === Load Trained CNN Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleASLCNN(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
model.eval()

# === Image Transform (must match training) ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

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

            # Get bounding box around the hand
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            # Clamp to frame size
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, w)
            y_max = min(y_max, h)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0 or hand_img.shape[0] < 10 or hand_img.shape[1] < 10:
                sign = "?"
            else:
                # Preprocess for CNN
                hand_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_pil = transforms.ToPILImage()(hand_rgb)
                input_tensor = transform(hand_pil).unsqueeze(0).to(device)

                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred = outputs.argmax(dim=1).item()
                    sign = inv_label_map.get(pred, "?")

            # Display prediction
            cv2.putText(frame, f"{label} Hand: {sign}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # Optionally, draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    else:
        cv2.putText(frame, "No hands detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("ASL CNN Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
