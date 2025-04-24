import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the CNN Model (same as your architecture)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Dropout(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 128),  # For 32x32 input
            nn.Tanh(),
            nn.Dropout(0.4),
            nn.Linear(128, 35),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Adjust input channel
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the model
model = CNNModel()
model.load_state_dict(torch.load("handwritten_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define transform (should match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Adjust for model's permute later
])

# Define label mapping
index_to_char = {i: chr(65 + i) for i in range(26)}  # A-Z
index_to_char.update({26+i: str(i) for i in range(9)})  # 0-8
index_to_char[35-1] = '9'  # Fix last class

# Streamlit UI
st.title("📝 Handwritten Character Recognition")
uploaded_file = st.file_uploader("Upload a handwritten character image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=False, width=200)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)  # [1, H, W, C]

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        pred_char = index_to_char.get(pred_class, "Unknown")

    st.success(f"✅ Predicted Character: **{pred_char}**")
