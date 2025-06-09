import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseRegressor(nn.Module):
    def __init__(self, base_model):
        super(SiameseRegressor, self).__init__()
        self.base = base_model
        self.base.fc = nn.Identity()
        self.similarity_head = nn.Sequential(
            nn.Linear(self.base.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.base(x)

    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.unsqueeze(1)

@st.cache_resource
def load_model():
    base = timm.create_model("xception", pretrained=False)
    base.conv1 = nn.Conv2d(1, base.conv1.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
    model = SiameseRegressor(base)
    model.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/siamese_similarity_model (1).pth", map_location=device))
    model.eval()
    return model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(img):
    img = img.convert("L")
    return transform(img).unsqueeze(0).to(device)

st.title("🖊️ Signature Similarity Checker")
img1_file = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
img2_file = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)
    st.image([img1, img2], caption=["Image 1", "Image 2"], width=150)

    model = load_model()
    with torch.no_grad():
        t1 = preprocess(img1)
        t2 = preprocess(img2)
        score = model(t1, t2).item()
        dissim = 1 - score

    if score > 0.8:
        verdict = "🟢 Highly Similar - Likely Genuine"
    elif score > 0.6:
        verdict = "🟡 Moderately Similar - Could be Genuine"
    elif score > 0.4:
        verdict = "🟠 Low Similarity - Possibly Forged"
    else:
        verdict = "🔴 Highly Dissimilar - Likely Forged"

    st.metric("Similarity Score", f"{score*100:.2f}%")
    st.metric("Dissimilarity", f"{dissim*100:.2f}%")
    st.success(f"Verdict: {verdict}")
