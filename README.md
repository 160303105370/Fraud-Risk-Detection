# 🛡️ Fraud Detection & Customer Risk Scoring (Hybrid Autoencoder + XGBoost)

### 📍 Overview  
This project builds a **hybrid machine learning system** to detect fraudulent financial transactions and score customer risk in real time.  
The goal was to **reduce false positives** — legitimate users incorrectly flagged as fraud — while maintaining strong fraud recall.  

By combining **unsupervised anomaly detection (Autoencoder)** with **supervised learning (XGBoost)**, the system learns to identify subtle, unusual patterns that traditional models often miss.

---

## 💼 Business Value

- Fraud costs financial institutions billions in losses each year, but overly aggressive fraud filters also hurt genuine customers.  
- This project delivers a **balanced approach**:
  - Fewer false alerts → improved **customer experience**  
  - Steady fraud recall → strong **protection and compliance**  
- Achieved a **~17% reduction in false positives** without losing recall, translating into faster fraud resolution and happier users.

---

## 🧠 Technical Summary

| Component | Description |
|------------|-------------|
| **Dataset** | [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 labeled as fraud (≈0.17%) |
| **Problem Type** | Binary classification (Fraud vs Legitimate) on highly imbalanced data |
| **Architecture** | 1️⃣ Train Autoencoder (AE) on non-fraud data only to learn “normal” transaction patterns.<br>2️⃣ Compute reconstruction error → add as new feature.<br>3️⃣ Train XGBoost model on augmented data for supervised classification. |
| **Goal Metric** | Reduce false positives while maintaining recall. Main metrics: Precision, Recall, PR-AUC. |

---

## ⚙️ Pipeline Steps

1. **Data Setup & EDA**  
   - Load dataset → check imbalance, missing values, and scaling needs.  
   - Create smaller dev sample for experimentation.

2. **Autoencoder Training (Unsupervised)**  
   - Input: normal transactions only (`Class = 0`).  
   - Architecture: Dense layers 64 → 32 → bottleneck (8) → 32 → 64.  
   - Output: reconstructed transaction.  
   - Loss: Mean Squared Error (MSE).  
   - Reconstruction error = anomaly score.

3. **Hybrid Model (XGBoost)**  
   - Input: all original features + `recon_error`.  
   - Handles imbalance via `scale_pos_weight`.  
   - Output: fraud probability score.  

4. **Evaluation & Metrics**  
   - Compare baseline GBM vs Hybrid model.  
   - Use PR-AUC, precision, recall, and confusion matrices.  
   - Quantify change in false positives.

---

## 📈 Results Summary

| Metric | Baseline GBM | Hybrid AE + XGBoost | Change |
|---------|---------------|----------------------|--------|
| True Positives | 105 | 105 | — |
| False Positives | 6 | 5 | ↓ **16.7%** |
| False Negatives | 18 | 18 | — |
| Precision | 0.946 | 0.955 | ↑ |
| Recall | 0.854 | 0.854 | — |
| PR-AUC | 0.94 | 0.96 | ↑ |

✅ **Key Takeaway:**  
The hybrid model preserved recall while reducing false positives by ~17%, improving overall fraud detection precision.

---


