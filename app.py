import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import imutils
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib

# Model definition
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
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load model
model = CNNModel().to('cuda')
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load label encoder
le = joblib.load("label_encoder.pkl")

# Utility functions
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method in ["right-to-left", "bottom-to-top"]:
        reverse = True
    if method in ["top-to-bottom", "bottom-to-top"]:
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def get_letters(img_path):
    letters = []
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 32, 32, 1)
            images_tensor = torch.tensor(thresh, dtype=torch.float32).to('cuda')
            ypred = [np.argmax(model(images_tensor).cpu().detach().numpy().flatten())]
            ypred = le.inverse_transform(ypred)
            [predicted] = ypred
            letters.append(predicted)
    return letters, image

def get_word(letter_list):
    return "".join(letter_list)

# Streamlit app
st.set_page_config(page_title="Handwritten Text OCR", layout="centered")
st.title("🖊️ Handwritten Text to Word Prediction")

uploaded_file = st.file_uploader("Upload an image of handwritten word", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())
    letters, processed_image = get_letters("temp_image.jpg")
    word = get_word(letters)

    st.image(processed_image, channels="BGR", caption="Processed Image with Detected Letters")
    st.success(f"✅ Predicted Word: **{word}**")
