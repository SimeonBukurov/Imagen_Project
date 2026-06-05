import os
import math
import cv2
import numpy as np
import torch
import mediapipe as mp
import tkinter as tk
from model import MultitaskFaceModel, get_device

# --- 1. CONFIGURATION & SETUP ---
MODEL_PATH = r"D:\Imagen Classes UPV\Project\models\custom_face_model_v1.pth"
WINDOW_NAME = 'Advanced Face AI Dashboard'
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
HAIR_LABELS = ['Black', 'Blond', 'Brown', 'Gray']

print("Loading custom PyTorch AI model...")
device = get_device()
model = MultitaskFaceModel().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 2. MEDIAPIPE SETUP ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE, 
    num_faces=3, # Upgraded to track up to 3 people at once!
    min_face_detection_confidence=0.6
)
landmarker = FaceLandmarker.create_from_options(options)
OVAL_INDICES = {idx for connection in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL for idx in (connection.start, connection.end)}

# --- 3. THE CENTROID TRACKER & STABILIZER ---

class PersonTracker:
    def __init__(self):
        self.next_id = 1
        self.people = {} # Format: {id: {"centroid": (x,y), "scores": [gender, hair, age], "lost_frames": 0}}
        
        # UPGRADE 1: Increased memory from 15 to 60 frames. 
        # It won't forget you instantly if you blink or look away!
        self.max_lost_frames = 60 

    def update(self, face_centroids):
        
        # UPGRADE 2: The Demo-Mode Reset. 
        # If the tracking memory is empty AND the camera sees no faces, reset the ID counter to 1.
        if len(self.people) == 0 and len(face_centroids) == 0:
            self.next_id = 1
            
        new_people = {}
        
        for centroid in face_centroids:
            matched_id = None
            min_dist = float('inf')
            
            # Find the closest existing person
            for pid, data in self.people.items():
                dist = math.hypot(centroid[0] - data["centroid"][0], centroid[1] - data["centroid"][1])
                if dist < 150 and dist < min_dist: 
                    min_dist = dist
                    matched_id = pid
                    
            if matched_id is not None:
                new_people[matched_id] = self.people[matched_id]
                new_people[matched_id]["centroid"] = centroid
                new_people[matched_id]["lost_frames"] = 0
                del self.people[matched_id] 
            else:
                # This is a new person!
                new_people[self.next_id] = {"centroid": centroid, "scores": None, "lost_frames": 0}
                matched_id = self.next_id
                self.next_id += 1

        # Keep lost people alive for a few frames just in case
        for pid, data in self.people.items():
            data["lost_frames"] += 1
            if data["lost_frames"] < self.max_lost_frames:
                new_people[pid] = data

        self.people = new_people
        return self.people

tracker = PersonTracker()

# --- 4. THE CYBER UI FUNCTION ---
def draw_cyber_hud(img, x_max, y_min, x_min, y_max, person_id, data):
    card_w, card_h = 220, 160
    hud_x = x_max + 40
    hud_y = y_min - 20
    
    # Keep HUD on screen
    if hud_x + card_w > img.shape[1]: hud_x = x_min - card_w - 40
    if hud_y < 0: hud_y = 20

    # Draw Targeting Line
    center_face_y = int((y_min + y_max) / 2)
    cv2.line(img, (x_max, center_face_y), (hud_x, hud_y + int(card_h/2)), (255, 200, 0), 1, cv2.LINE_AA)

    # Glassmorphism Box
    overlay = img.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + card_w, hud_y + card_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.line(img, (hud_x, hud_y), (hud_x, hud_y + card_h), (255, 200, 0), 3)

    # Header
    cv2.putText(img, f"PERSON {person_id}", (hud_x + 15, hud_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.line(img, (hud_x + 15, hud_y + 35), (hud_x + card_w - 15, hud_y + 35), (100, 100, 100), 1)

    # Data
    if data:
        text_y = hud_y + 60
        for title, value in data.items():
            cv2.putText(img, title, (hud_x + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(img, value, (hud_x + 15, text_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            text_y += 40

# --- 5. CENTER WINDOW & CAMERA ---
root = tk.Tk()
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
cv2.moveWindow(WINDOW_NAME, int((root.winfo_screenwidth() - TARGET_WIDTH) / 2), int((root.winfo_screenheight() - TARGET_HEIGHT) / 2))
root.destroy()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
print("Camera is live. Press 'q' to exit.")

frame_count = 0

# --- 6. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: continue
    
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect(mp_image)

    current_centroids = []
    face_boxes = []

    # 1. Map all faces in the frame
    if results.face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results.face_landmarks:
            x_coords = [int(lm.x * w) for lm in face_landmarks]
            y_coords = [int(lm.y * h) for lm in face_landmarks]
            
            # Draw Face Contour
            for idx, lm in enumerate(face_landmarks):
                if idx in OVAL_INDICES:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 200, 0), -1)

            margin = 30
            x_min, x_max = max(0, min(x_coords) - margin), min(w, max(x_coords) + margin)
            y_min, y_max = max(0, min(y_coords) - margin), min(h, max(y_coords) + margin)
            
            centroid = (int((x_min + x_max)/2), int((y_min + y_max)/2))
            current_centroids.append(centroid)
            face_boxes.append((x_min, x_max, y_min, y_max))

    # 2. Update the Tracker
    active_people = tracker.update(current_centroids)

    # 3. Process AI & Draw UI for each person
    for i, centroid in enumerate(current_centroids):
        # Find which ID belongs to this centroid
        person_id = None
        for pid, pdata in active_people.items():
            if pdata["centroid"] == centroid:
                person_id = pid
                break
                
        if not person_id: continue
        x_min, x_max, y_min, y_max = face_boxes[i]

        # AI Prediction Check
        if frame_count % 10 == 0 and x_max > x_min and y_max > y_min:
            try:
                face_crop = frame[y_min:y_max, x_min:x_max]
                crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                crop_normalized = (cv2.resize(crop_rgb, (224, 224)) / 127.5) - 1.0
                
                input_tensor = torch.tensor(crop_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    preds = model(input_tensor)
                    new_g, new_h, new_a = preds[0].item(), preds[1][0], preds[2].item()
                
                # --- STABILIZER LOGIC (Exponential Moving Average) ---
                if tracker.people[person_id]["scores"] is None:
                    tracker.people[person_id]["scores"] = [new_g, new_h, new_a]
                else:
                    old_g, old_h, old_a = tracker.people[person_id]["scores"]
                    # Smooth by keeping 70% of the old memory, 30% of the new frame
                    tracker.people[person_id]["scores"] = [
                        (old_g * 0.7) + (new_g * 0.3),
                        (old_h * 0.7) + (new_h * 0.3),
                        (old_a * 0.7) + (new_a * 0.3)
                    ]
            except Exception: pass

        # Draw the UI using stabilized scores
        display_data = None
        if tracker.people[person_id]["scores"]:
            g_score, h_score, a_score = tracker.people[person_id]["scores"]
            display_data = {
                "GENDER": "Male" if g_score > 0.5 else "Female",
                "AGE EST": "Young Adult" if a_score > 0.5 else "Mature",
                "HAIR": HAIR_LABELS[torch.argmax(h_score).item()]
            }
            
        draw_cyber_hud(frame, x_max, y_min, x_min, y_max, person_id, display_data)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()