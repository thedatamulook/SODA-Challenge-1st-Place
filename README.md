# 🏆 1st Place Solution  
A robust deep learning pipeline using **ConvNeXt-Base**, **Stratified 5-Fold CV**, **TTA**, and **Hard Majority Voting** to secure **1st place**.

## 📌 CCWin Challenges Series 1: SODA Challenge  
**Solution Writeup · 1st place · Jun 4, 2026**

## 👥 Team Information
**Team Name:** Data Minds  
**Department:** Data Science, College of Computing  
**Institution:** Umm Al-Qura University  

### Team Members
- Ghadof Abdulrahman Almoqbil — 446001462  
- Layan Mohammed Alzahrani — 446002694  
- **Mulook Faleh Al-Hudhali — 446002464**  
- Rana Mohammed Alzahrani — 446002816  

## 📘 Overview
The objective of this challenge was to classify Saudi organic date varieties from images. The dataset contained multiple date classes, making it a complex multi-class image classification problem. Our final pipeline achieved **1st place** on the Private Leaderboard using an **Ultimate Hard Majority Voting Ensemble** powered by **ConvNeXt-Base**. The pipeline combined stratified cross-validation, heavy GPU-accelerated augmentation, mixed-precision fine-tuning, Test-Time Augmentation (TTA), and a smart tie-breaker voting mechanism.

## 🛠️ Implementation Environment
### Frameworks
- TensorFlow & Keras  
- Keras Applications  
- Pandas & NumPy  
- Scikit-learn  

### Hardware
Training used **Kaggle Dual GPU T4x2** with:
- `tf.distribute.MirroredStrategy`  
- Mixed Precision (`mixed_float16`)  

**Image Size:** 320×320  
**Batch Size:** 16  
**Epochs:** 15–20  
**Learning Rate:** 2e-4  

## 🔧 Data Preprocessing & Validation Strategy
### Stratified 5-Fold CV
Ensured balanced class distribution and stable validation performance.

### GPU-Accelerated Augmentation
Included RandomFlip, RandomRotation, RandomZoom, and RandomContrast.

## 🚀 Training & Inference Strategy
### Fine-Tuning
Started from ImageNet weights, froze early blocks, trained top 60 layers, used AdamW optimizer and ReduceLROnPlateau.

### Test-Time Augmentation (TTA)
Each test image evaluated twice: original and horizontally flipped.

### Ultimate Majority Voting Ensemble
Combined predictions from three models:

| Model | Score |
|-------|--------|
| Champion Single Model | **0.90301** |
| Soft-Ensemble Pipeline | 0.89710 |
| Optimized Team Baseline | 0.89000 |

Tie-breaker rule: trust the highest-scoring model.

#### Majority Voting Code
```python
def get_majority(row):
    votes = [row['label1'], row['label2'], row['label3']]
    counts = {v: votes.count(v) for v in set(votes)}
    max_count = max(counts.values())
    if max_count >= 2:
        for k, v in counts.items():
            if v == max_count:
                return k
    else:
        return row['label1']


⚠️ Lessons Learned
Pseudo-labeling with confidence ≥ 0.98 caused data leakage and poisoning, reducing validation accuracy. It was discarded in favor of the clean ensemble.

🔗 Code Availability
Due to competition guidelines, the full training pipeline remains private. The core ensembling logic is shared above.

🌟 Acknowledgements
We thank the SODA Challenge organizers and Umm Al-Qura University for this opportunity to apply advanced deep learning techniques to a real-world Saudi agricultural dataset.

🏷️ Tags
Deep Learning • Computer Vision • Image Classification • Ensembling • Transfer Learning
