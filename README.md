# 🦁 Swin Animal Classifier

A professional, end-to-end Computer Vision project that brings the **Swin Transformer (Shifted Window Transformer)** from a research-level scratch implementation to a production-ready web application.

## 🚀 Live Application
Try the live model here: **[Swin Animal Classifier Live](https://cjcedsqdxffm74zhoo9jxv.streamlit.app)**

---

## 📌 Project Overview
This project explores the **Swin Transformer** architecture, a hierarchical Vision Transformer that uses **Shifted Windows** to solve the high-computational cost of global self-attention. 

### 🌟 Key Features
- **Hierarchical Architecture:** Captures both local textures and global shapes.
- **Shifted Window Attention:** Enables cross-window communication for better accuracy.
- **Production Ready:** Optimized for CPU-based inference on Streamlit Cloud.
- **User-Friendly Interface:** Simple drag-and-drop image classification.

---

## 🛠️ How It Works (The Math)
1. **Patch Partitioning:** Breaks the input image into $4 \times 4$ small patches.
2. **Linear Embedding:** Projects these patches into a high-dimensional feature space.
3. **W-MSA & SW-MSA:** Computes attention within local windows and then shifts those windows to allow information to flow across boundaries.
4. **Patch Merging:** Reduces spatial resolution while doubling feature depth (similar to CNN pooling).
5. **Softmax Head:** Outputs the probability of 1,000 different categories.

---

## 🧪 Research & Development (Google Colab)
The project started with a "build-from-scratch" approach where I coded the `PatchEmbed`, `WindowAttention`, and `SwinBlock` layers manually. 

- **Scratch Project Link:** [Open Google Colab Notebook](https://colab.research.google.com/drive/1KBj-qyv0nRVBSuejVMTbuRlbiM00_UKa)
- **Dataset:** Initially tested on **CIFAR-100** with custom training loops and `OneCycleLR` scheduling.
- **Optimizer:** AdamW with weight decay.

---

## 💻 Setup & Local Execution
Want to run this on your own PC? Follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com
cd swin-animal-classifier
