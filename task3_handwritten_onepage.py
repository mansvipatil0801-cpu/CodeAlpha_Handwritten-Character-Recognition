"""
CodeAlpha ML Internship - TASK 3: Handwritten Character Recognition
Dataset   : sklearn digits (REAL, 1797 handwritten 0-9 images, 8x8px) - same family
            as MNIST. This sandbox has no internet, so full MNIST/EMNIST (which
            need a download) aren't reachable; swap-in loaders for both are below.
Model     : Deep neural network (MLP) as an offline CNN-equivalent (no internet
            here to install tensorflow/torch). Real Conv2D CNN code is included
            as a comment for when you run this locally/Colab.
Run: python3 task3_handwritten_recognition.py
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RNG = 42

# ---------- REAL dataset (digits 0-9, 8x8 images) ----------
digits = load_digits()
X, y, images = digits.images.reshape(len(digits.images), -1), digits.target, digits.images
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} pixel features, {len(np.unique(y))} classes")

# To use FULL MNIST (28x28, 70,000 images) or EMNIST (letters+digits), once online:
#   from tensorflow.keras.datasets import mnist
#   (Xtr, ytr), (Xte, yte) = mnist.load_data()
#   # then feed into a real Conv2D CNN, e.g.:
#   model = keras.Sequential([
#       keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
#       keras.layers.MaxPooling2D(2),
#       keras.layers.Conv2D(64, 3, activation='relu'),
#       keras.layers.Flatten(),
#       keras.layers.Dense(128, activation='relu'),
#       keras.layers.Dense(10, activation='softmax')])
#   # For EMNIST (letters), just change the final Dense to len(classes) (26 or 47).
#   # For full words/sentences, extend to a CRNN: CNN feature extractor -> BiLSTM -> CTC loss.

Xtr, Xte, ytr, yte, img_tr, img_te = train_test_split(X, y, images, test_size=0.2, stratify=y, random_state=RNG)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                       alpha=1e-4, max_iter=300, random_state=RNG)
print("Training deep neural network...")
model.fit(Xtr_s, ytr)
pred = model.predict(Xte_s)
acc = accuracy_score(yte, pred)
cv_acc = cross_val_score(model, scaler.transform(X), y, cv=5).mean()

print(f"\nTest Accuracy: {acc*100:.2f}%  |  5-Fold CV Accuracy: {cv_acc*100:.2f}%\n")
print(classification_report(yte, pred))

# Confusion matrix
cm = confusion_matrix(yte, pred)
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - Handwritten Digit Recognition (Acc: {acc*100:.1f}%)")
plt.tight_layout(); plt.savefig("task3_confusion_matrix.png", dpi=150)
print("Saved: task3_confusion_matrix.png")

# Sample predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(img_te[i], cmap="gray_r")
    ax.set_title(f"True:{yte[i]} Pred:{pred[i]}", color="green" if pred[i] == yte[i] else "red", fontsize=10)
    ax.axis("off")
plt.suptitle("Sample Predictions (green=correct, red=wrong)")
plt.tight_layout(); plt.savefig("task3_sample_predictions.png", dpi=150)
print("Saved: task3_sample_predictions.png")
