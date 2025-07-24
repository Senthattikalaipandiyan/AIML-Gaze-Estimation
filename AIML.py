# gaze_estimation.py

# --------------------- PACKAGE IMPORT ---------------------
import os
import cv2
import time
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

print("✅ Imported Packages Successfully")


# --------------------- LOAD IMAGE DATA ---------------------
folder_path = "MPIIFaceGaze_preprocessed/Image/p00/face"
images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
image_data = []

for img_name in images:
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    image_data.append(img)

image_data = np.array(image_data)
print(f"✅ Loaded {len(image_data)} images")

# --------------------- SPLIT DATA ---------------------
train_images, test_images = train_test_split(image_data, test_size=0.2, random_state=42)
train_images = train_images / 255.0
test_images = test_images / 255.0
print(f"✅ Train Images: {len(train_images)}, Test Images: {len(test_images)}")


# --------------------- CUSTOM DATASET ---------------------
class FaceDataset(Dataset):
    def __init__(self, face_images, left_eye_images, right_eye_images):
        self.face_images = face_images
        self.left_eye_images = left_eye_images
        self.right_eye_images = right_eye_images
        self.gaze_points = torch.randn(len(face_images), 2)  # Dummy targets
        self.gaze_directions = torch.randn(len(face_images), 2)

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        def to_tensor(img):
            return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        face_img = to_tensor(self.face_images[idx])
        left_eye_img = to_tensor(self.left_eye_images[idx])
        right_eye_img = to_tensor(self.right_eye_images[idx])
        gaze_point = self.gaze_points[idx]
        gaze_direction = self.gaze_directions[idx]
        return face_img, left_eye_img, right_eye_img, gaze_point, gaze_direction


# Dummy data generation for training
train_face_images = np.random.rand(100, 64, 64, 3)
train_left_eyes = np.random.rand(100, 64, 64, 3)
train_right_eyes = np.random.rand(100, 64, 64, 3)

test_face_images = np.random.rand(20, 64, 64, 3)
test_left_eyes = np.random.rand(20, 64, 64, 3)
test_right_eyes = np.random.rand(20, 64, 64, 3)

train_dataset = FaceDataset(train_face_images, train_left_eyes, train_right_eyes)
test_dataset = FaceDataset(test_face_images, test_left_eyes, test_right_eyes)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# --------------------- ATTENTION MODULE ---------------------
class FocusAttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(FocusAttentionLayer, self).__init__()
        self.fc_face = nn.Linear(feature_dim, feature_dim)
        self.fc_eye = nn.Linear(feature_dim, feature_dim)

    def forward(self, face_feat, eye_feat):
        face_proj = self.fc_face(face_feat)
        eye_proj = self.fc_eye(eye_feat)
        attention_score = torch.sum(face_proj * eye_proj, dim=1, keepdim=True)
        attention_weight = torch.sigmoid(attention_score)
        attended_feat = attention_weight * eye_feat + (1 - attention_weight) * face_feat
        return attended_feat


# --------------------- MULTI-TASK GAZE NETWORK ---------------------
class GazeMultitaskNet(nn.Module):
    def __init__(self):
        super(GazeMultitaskNet, self).__init__()
        self.output_size = 64
        self.feature_map_size = 7
        self.flatten_dim = self.output_size * self.feature_map_size * self.feature_map_size

        self.conv_face = self._conv_block()
        self.conv_left_eye = self._conv_block()
        self.conv_right_eye = self._conv_block()

        self.pool = nn.AdaptiveAvgPool2d((self.feature_map_size, self.feature_map_size))
        self.attn = FocusAttentionLayer(self.flatten_dim)
        self.fc = nn.Linear(self.flatten_dim, 128)

        self.gaze_point_head = nn.Linear(128, 2)
        self.gaze_direction_head = nn.Linear(128, 2)

    def _conv_block(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

    def forward(self, face_img, left_eye_img, right_eye_img):
        face_feat = self.pool(self.conv_face(face_img))
        left_feat = self.pool(self.conv_left_eye(left_eye_img))
        right_feat = self.pool(self.conv_right_eye(right_eye_img))

        face_feat = face_feat.view(face_feat.size(0), -1)
        left_feat = left_feat.view(left_feat.size(0), -1)
        right_feat = right_feat.view(right_feat.size(0), -1)

        eye_feat = (left_feat + right_feat) / 2
        attended_feat = self.attn(face_feat, eye_feat)

        x = F.relu(self.fc(attended_feat))
        gaze_point = self.gaze_point_head(x)
        gaze_direction = self.gaze_direction_head(x)

        return gaze_point, gaze_direction


# --------------------- TRAINING ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeMultitaskNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"✅ GazeMultitaskNet initialized on {device}")
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for face_img, left_eye_img, right_eye_img, target_point, target_direction in train_loader:
        face_img = face_img.to(device)
        left_eye_img = left_eye_img.to(device)
        right_eye_img = right_eye_img.to(device)
        target_point = target_point.to(device)
        target_direction = target_direction.to(device)

        pred_point, pred_direction = model(face_img, left_eye_img, right_eye_img)

        loss_point = criterion(pred_point, target_point)
        loss_direction = criterion(pred_direction, target_direction)
        loss = loss_point + loss_direction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

print("✅ Training Completed Successfully")

# --------------------- SAVE MODEL ---------------------
model_path = "gaze_model.pth"
torch.save(model.state_dict(), model_path)
print("✅ Model saved successfully to gaze_model.pth")


# --------------------- REAL-TIME OPENCV INFERENCE ---------------------
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
last_logged_time = time.time()


def preprocess_roi(roi):
    roi = cv2.resize(roi, (64, 64)).astype(np.float32) / 255.0
    return torch.tensor(roi).permute(2, 0, 1).unsqueeze(0).to(device)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = eyes[0]
            ex2, ey2, ew2, eh2 = eyes[1]

            left_eye = frame[y+ey1:y+ey1+eh1, x+ex1:x+ex1+ew1]
            right_eye = frame[y+ey2:y+ey2+eh2, x+ex2:x+ex2+ew2]

            face_tensor = preprocess_roi(face_roi)
            left_tensor = preprocess_roi(left_eye)
            right_tensor = preprocess_roi(right_eye)

            with torch.no_grad():
                gaze_point, gaze_dir = model(face_tensor, left_tensor, right_tensor)
                gaze_point = gaze_point.squeeze().cpu().numpy()
                gaze_dir = gaze_dir.squeeze().cpu().numpy()

            center_x, center_y = int(x + w / 2), int(y + h / 2)
            point_x = int(center_x + gaze_point[0] * 50)
            point_y = int(center_y + gaze_point[1] * 50)
            dir_x = int(center_x + gaze_dir[0] * 50)
            dir_y = int(center_y + gaze_dir[1] * 50)

            cv2.arrowedLine(frame, (center_x, center_y), (dir_x, dir_y), (0, 255, 0), 2)
            cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            left_eye_center = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
            right_eye_center = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)

            current_time = time.time()
            if current_time - last_logged_time >= 1:
                print(f"👁 Left Eye: X={left_eye_center[0]} Y={left_eye_center[1]}")
                print(f"👁 Right Eye: X={right_eye_center[0]} Y={right_eye_center[1]}")
                last_logged_time = current_time

            white_screen = np.ones_like(frame) * 255
            cv2.circle(white_screen, left_eye_center, 5, (0, 0, 255), -1)
            cv2.circle(white_screen, right_eye_center, 5, (255, 0, 0), -1)
            combined = np.hstack((frame, white_screen))

            cv2.imshow('Gaze Estimation UI (Left) & Coordinates (Right)', combined)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Real-Time Gaze Estimation Completed")
