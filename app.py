import streamlit as st
from PIL import Image
from model_utils import load_swin_model, predict_pil
import torch

st.set_page_config(page_title="Swin Classifier", page_icon="🦁")

st.title("🦁 Swin Transformer: Image Classifier")
st.write("Swin Transformer ব্যবহার করে যেকোনো ছবি শনাক্ত করুন।")

@st.cache_resource
def get_model():
    return load_swin_model()

model, device = get_model()

uploaded_file = st.file_uploader("একটি ছবি আপলোড করুন...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='আপনার আপলোড করা ছবি', use_column_width=True)
    
    with st.spinner("মডেলটি ছবিটি বিশ্লেষণ করছে..."):
        results = predict_pil(image, model, device)
        
        top_result = results[0]
        
        st.success(f"**সনাক্তকৃত বস্তু:** {top_result['label']}")
        st.info(f"**নিশ্চয়তা (Confidence):** {top_result['probability']*100:.2f}%")