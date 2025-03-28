# 🤟 American Sign Language Image Classification – Deep Learning with CNNs

This project explores deep learning methods to classify American Sign Language (ASL) alphabet images using dense and convolutional neural networks (CNNs). Developed for the "Using Machine Learning Tools" (COMP SCI 7317) course, the goal was to build a lightweight, accurate model within strict resource constraints suitable for mobile deployment.

---

## 📁 Dataset

- 32x32 grayscale images of ASL alphabets (A–Z, excluding J and Z)
- 25 classes, pre-split into training and test sets
- Images are centered and pre-labeled

---

## 🎯 Objectives

- Preprocess image data and visualize class distributions  
- Build and evaluate both **Dense** and **CNN** models  
- Tune hyperparameters (layers, learning rate, batch size, optimizer)  
- Select a final model under constraints:
  - ≤ 50 training runs
  - ≤ 500,000 total parameters

---

## 🛠 Tools & Libraries

- Python (Jupyter Notebook)  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn

---

## 🔄 Preprocessing

- Checked for missing values  
- Normalized pixel values  
- Encoded labels using one-hot encoding  
- Reshaped input arrays for CNN compatibility

---

## 🧪 Models Built

### 🔹 Dense Neural Network (Baseline)
- Architecture: [16, 32, 64]
- Optimizer: Adamax  
- Activation: ReLU  
- Test Accuracy: **~90.2%**

### 🔸 Convolutional Neural Network (Final Model)
- Architecture: Conv2D layers with [16, 32, 64, 16] filters  
- Activation: ELU  
- Optimizer: Adamax  
- Epochs: 11  
- Batch size: 32  
- Test Accuracy: **~94.4%**

---

## 📊 Evaluation Metrics

- Classification Report  
- Confusion Matrix  
- Class-wise Accuracy  
- Misclassified Class Tracking

### 🔍 Highlights
| Metric            | Value       |
|-------------------|-------------|
| Best Model        | CNN         |
| Test Accuracy     | 94.4%       |
| Most Accurate     | 'W'         |
| Least Accurate    | 'Q'         |
| Most Misclassified| 'E' as 'F'  |

---

## 📈 Visualizations

- Confusion matrix heatmap  
- Accuracy per class  
- Loss and accuracy curves during training  
- Predicted vs actual class images

---

## ✅ Recommendation

CNN architecture using Adamax and ELU activation meets constraints and yields high test accuracy. Suitable for lightweight deployment on mobile platforms with potential improvements through dropout and further hyperparameter tuning.

---

## 📁 Files

- `UMLT_ASSIGNMENT_3_A1899824.ipynb` – Full Jupyter Notebook  
- `UMLT_ASSIGNMENT_3_A1899824.pdf` – Final report  
- `UMLT_ASSIGNMENT_3_A1899824.py` – Script version (optional)

---

## 👤 Author

**Aditya Venugopalan Nediyirippil**  
*University of Adelaide – COMP SCI 7317 (UMLT)*  
GitHub: [a1899824-aditya](https://github.com/a1899824-aditya)

