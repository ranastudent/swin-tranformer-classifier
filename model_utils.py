import torch
import timm
import torchvision.transforms as transforms
import urllib.request
import json

# -----------------------------
# Labels
# -----------------------------
def load_labels():
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = urllib.request.urlopen(url)
        data = response.read().decode("utf-8").splitlines()
        
        return {i: name for i, name in enumerate(data)}
    
    except Exception as e:
        print("Label load error:", e)
        return {i: f"Category {i}" for i in range(1000)}

LABELS = load_labels()


# -----------------------------
# Model
# -----------------------------
def load_swin_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    model.eval()
    model.to(device)
    
    return model, device


# -----------------------------
# Transform
# -----------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# -----------------------------
# Prediction (PIL Image)
# -----------------------------
def predict_pil(image, model, device, topk=5):
    transform = get_transform()
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
    
    top_probs, top_idxs = torch.topk(probs, topk)
    
    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        label = LABELS[idx.item()]
        results.append({
            "label": label.replace("_", " ").title(),
            "probability": prob.item()
        })
    
    return results