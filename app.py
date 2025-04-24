import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import imutils
from PIL import Image
import os

# ---------------------- Device Setup ---------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- Model Definition ---------------------- #
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(128, 35),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # -> (batch, C, H, W)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ---------------------- Classes List ---------------------- #
classes = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','P','Q','R','S','T','U',
    'V','W','X','Y','Z'
]

# ---------------------- Load Model ---------------------- #
@st.cache_resource
def load_model():
    model = CNNModel().to(device)
    model.load_state_dict(torch.load("handwritten_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ---------------------- Utility Functions ---------------------- #
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    axis = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        axis = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boundingBoxes = zip(
        *sorted(
            zip(cnts, boundingBoxes),
            key=lambda b: b[1][axis], reverse=reverse
        )
    )
    return cnts

def get_letters(image_path):
    letters = []
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # invert & threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh, None, iterations=2)

    # find contours
    cnts = cv2.findContours(
        dilated.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")

    for c in cnts:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            # draw box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
            # extract ROI
            roi = gray[y:y+h, x:x+w]
            roi = cv2.threshold(
                roi, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_CUBIC)
            roi = roi.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=-1)       # HWC
            roi = roi.reshape(1, 32, 32, 1)           # batch,H,W,C
            tensor = torch.tensor(roi, dtype=torch.float32).to(device)
            # predict
            with torch.no_grad():
                preds = model(tensor)
                idx = torch.argmax(preds, dim=1).item()
            char = classes[idx]  # Direct mapping to character
            letters.append(char)

    return letters, image

def get_word(letter_list):
    return "".join(letter_list)

# ---------------------- Streamlit App ---------------------- #
st.set_page_config(page_title="Handwritten OCR", layout="centered")
st.title("✍️ Handwritten Text Recognition (OCR)")
st.write("Upload an image of a handwritten word; the app will segment and recognize each character.")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded:
    # save temp
    tmp_path = "temp_input.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # process & predict
    letters, processed_img = get_letters(tmp_path)
    word = get_word(letters)

    # display
    st.image(processed_img, channels="BGR",
             caption="Segmented Characters", use_column_width=True)
    st.success(f"✅ Predicted Word: **{word}**")

    # cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
