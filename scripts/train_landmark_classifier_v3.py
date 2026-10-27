"""
train_landmark_classifier_v3.py
================================
Perbaikan dari v2 — mengatasi masalah Y over-prediction.

Root cause v2:
- label_smoothing=0.1 membuat model terlalu "ragu" dan memilih Y sebagai fallback
- Normalisasi wrist-relative kehilangan informasi spatial antar 2 tangan

Perbaikan v3:
1. StandardScaler (fit on training data, saved as pickle) — normalisasi yang stabil
2. Hapus label_smoothing — pakai sparse_categorical_crossentropy biasa
3. Arsitektur lebih baik + BatchNorm + Dropout
4. Augmentasi hanya noise kecil (tidak ada translate — mencegah Y artifacts)
5. Class weights untuk class yang sulit
6. Simpan scaler bersama model untuk dipakai di detector
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)

DATA_PATH        = os.path.join(BASE_DIR, 'data', 'bisindo_Landmarks.csv')
MODEL_SAVE_PATH  = os.path.join(BASE_DIR, 'models', 'bisindo_landmark_classifier.h5')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'landmark_scaler.pkl')
DICTIONARY_DIR   = os.path.join(BASE_DIR, 'dictionary')
RESULTS_DIR      = os.path.join(BASE_DIR, 'results')

# --- Hyperparameter ---
NUM_EPOCHS   = 200
BATCH_SIZE   = 32    # Lebih kecil = update lebih sering = generalisasi lebih baik
NOISE_STD    = 0.008 # Augmentasi: Gaussian noise ringan
AUGMENT_MULT = 2     # 2x data (lebih konservatif dari v2)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(DICTIONARY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# NORMALISASI WRIST-RELATIVE (per tangan, lebih robust)
# =============================================================================
def normalize_landmarks_v3(landmarks_flat):
    """
    Normalisasi per-hand wrist-relative + scale-invariant.
    Untuk tangan yang tidak ada, tetap dijaga sebagai zeros agar
    model bisa membedakan one-handed vs two-handed gestures.
    """
    result = []
    for hand_start in [0, 63]:
        hand = np.array(landmarks_flat[hand_start:hand_start + 63],
                        dtype=np.float32).reshape(21, 3)

        if np.allclose(hand, 0, atol=1e-7):
            result.extend([0.0] * 63)
            continue

        # Wrist-relative translation
        wrist = hand[0].copy()
        hand = hand - wrist

        # Scale normalisasi menggunakan jarak wrist -> landmark 9 (middle MCP)
        scale = np.linalg.norm(hand[9])
        if scale > 1e-6:
            hand = hand / scale

        result.extend(hand.flatten().tolist())
    return result


def normalize_batch_v3(X):
    return np.array([normalize_landmarks_v3(row) for row in X], dtype=np.float32)


# =============================================================================
# AUGMENTASI (hanya noise, tidak ada translate)
# =============================================================================
def augment_with_noise(X, y, augment_mult=2, noise_std=0.008):
    """Augmentasi ringan hanya dengan Gaussian noise."""
    X_list = [X]
    y_list = [y]
    for _ in range(augment_mult):
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        # Jangan tambahkan noise ke kolom yang nol (tangan tidak ada)
        mask = (X != 0).astype(np.float32)
        X_noisy = X + noise * mask
        X_list.append(X_noisy)
        y_list.append(y)
    X_aug = np.vstack(X_list)
    y_aug = np.concatenate(y_list)
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# =============================================================================
# MAIN TRAINING
# =============================================================================
print("=" * 60)
print("  Training Model BISINDO v3 (Fixed Y Over-Prediction)")
print("=" * 60)

# --- 1. Load Data ---
print(f"\n[1/7] Memuat data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X_raw = df.drop('label', axis=1).values.astype(np.float32)
y_labels = df['label'].values
print(f"      {df.shape[0]} sampel, {df['label'].nunique()} kelas")

# --- 2. Encode Label ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

class_names_path = os.path.join(DICTIONARY_DIR, 'class_names_landmark.txt')
with open(class_names_path, 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")
print(f"      Kelas: {list(class_names)}")

# --- 3. Normalisasi Wrist-Relative ---
print(f"\n[2/7] Normalisasi wrist-relative...")
X_norm = normalize_batch_v3(X_raw)
print(f"      Selesai. Shape: {X_norm.shape}")

# --- 4. Split Train/Val ---
X_train, X_val, y_train, y_val = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y_labels
)
print(f"\n[3/7] Split: train={len(X_train)}, val={len(X_val)}")

# --- 5. StandardScaler (fit pada train saja, transform train & val) ---
print(f"\n[4/7] StandardScaler (fit on train)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)

# Simpan scaler — WAJIB dipakai juga di detector
with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"      Scaler disimpan: {SCALER_SAVE_PATH}")

# --- 6. Augmentasi ---
print(f"\n[5/7] Augmentasi noise (x{AUGMENT_MULT})...")
X_train_aug, y_train_aug = augment_with_noise(
    X_train_scaled, y_train, augment_mult=AUGMENT_MULT, noise_std=NOISE_STD
)
print(f"      Data training: {len(X_train_aug)} sampel")

# --- 7. Class Weights ---
cw_values = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weight_dict = dict(enumerate(cw_values))

# --- 8. Bangun Model ---
print(f"\n[6/7] Membangun model v3...")
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.30),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.20),

    Dense(num_classes, activation='softmax'),
], name='bisindo_mlp_v3')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',   # Tidak pakai label smoothing
    metrics=['accuracy']
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                      min_lr=1e-6, verbose=1),
]

# --- 9. Training ---
print(f"\n[7/7] Mulai training ({NUM_EPOCHS} epoch maks)...")
start = time.time()
history = model.fit(
    X_train_aug, y_train_aug,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_scaled, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)
print(f"\nTraining selesai: {time.time()-start:.1f}s")

# =============================================================================
# EVALUASI
# =============================================================================
print("\n" + "=" * 60)
print("  EVALUASI MODEL v3")
print("=" * 60)

model_best = tf.keras.models.load_model(MODEL_SAVE_PATH)
y_pred_proba = model_best.predict(X_val_scaled, verbose=0)
y_pred = y_pred_proba.argmax(axis=1)

report = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)
print(report)

# Simpan report
report_path = os.path.join(RESULTS_DIR, 'classification_report_v3.txt')
with open(report_path, 'w') as f:
    f.write("Model v3 - Classification Report\n" + "="*50 + "\n" + report)
print(f"Report: {report_path}")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(14, 12))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='lightgray', ax=ax)
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title('Confusion Matrix - Model BISINDO v3', fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_v3.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrix: {cm_path}")

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'], label='Train', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val', color='tomato')
axes[0].set_title('Akurasi', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Akurasi')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val', color='tomato')
axes[1].set_title('Loss', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'training_history_v3.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"History plot: {plot_path}")

val_loss, val_acc = model_best.evaluate(X_val_scaled, y_val, verbose=0)
print("\n" + "=" * 60)
print(f"  HASIL AKHIR v3")
print("=" * 60)
print(f"  Val Accuracy : {val_acc*100:.2f}%")
print(f"  Val Loss     : {val_loss:.4f}")
print(f"  Model       : {MODEL_SAVE_PATH}")
print(f"  Scaler      : {SCALER_SAVE_PATH}")
print("=" * 60)
print("\nSelesai! Update realtime_detector_mediapipe.py untuk pakai scaler baru.")
