
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SiameseRegressor(nn.Module):
    def __init__(self, base_model):
        super(SiameseRegressor, self).__init__()
        self.base = base_model
        self.base.fc = nn.Identity()
    
    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.unsqueeze(1)

def load_model(path):
    base = timm.create_model("xception", pretrained=True)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        base.conv1.weight[:, 0] = base.conv1.weight.mean(dim=1)
    model = SiameseRegressor(base)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def get_color_and_comment(dissimilarity):
    if dissimilarity < 0.2:
        return "#00cc66", "✅ Signatures are very similar (Likely Genuine)"
    elif dissimilarity < 0.4:
        return "#ffcc00", "🟡 Somewhat similar"
    elif dissimilarity < 0.6:
        return "#ff9933", "🟠 Low similarity (Could be forged)"
    else:
        return "#cc0000", "❌ Highly dissimilar (Likely Forged)"

st.set_page_config(page_title="Signature Verifier", layout="centered")
st.title("🖊️ Signature Verification App")
st.markdown("Upload two signature images to compare similarity.")

uploaded1 = st.file_uploader("Upload Signature 1", type=["png", "jpg", "jpeg"])
uploaded2 = st.file_uploader("Upload Signature 2", type=["png", "jpg", "jpeg"])

if uploaded1 and uploaded2:
    img1 = Image.open(uploaded1).convert("L")
    img2 = Image.open(uploaded2).convert("L")
    
    input1 = transform(img1).unsqueeze(0).to(device)
    input2 = transform(img2).unsqueeze(0).to(device)

    model = load_model("model/siamese_similarity_model.pth")

    with torch.no_grad():
        similarity = model(input1, input2).item()

    dissimilarity = 1 - similarity
    color, comment = get_color_and_comment(dissimilarity)

    st.image([img1, img2], caption=["Signature 1", "Signature 2"], width=150)
    st.markdown(f"### 🔍 Dissimilarity: **{dissimilarity * 100:.2f}%**")
    st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:10px;text-align:center;color:white;font-weight:bold;'>{comment}</div>", unsafe_allow_html=True)
