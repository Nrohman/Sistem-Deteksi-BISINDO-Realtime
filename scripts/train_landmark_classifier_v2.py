"""
train_landmark_classifier_v2.py
================================
Versi perbaikan model BISINDO dengan:
1. Normalisasi landmark wrist-relative + scale-invariant
2. Arsitektur lebih dalam + BatchNormalization
3. Data augmentation in-memory (noise, scale, translate)
4. Label smoothing untuk mengurangi overconfidence
5. Learning rate scheduler

Perbaikan ini mengatasi masalah:
- A kedeteksi sebagai Y
- K kedeteksi sebagai P
- Beberapa huruf tidak terdeteksi sama sekali
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend agar bisa save tanpa display
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Konfigurasi Path (relatif terhadap lokasi script) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH        = os.path.join(BASE_DIR, 'data', 'bisindo_Landmarks.csv')
MODEL_SAVE_PATH  = os.path.join(BASE_DIR, 'models', 'bisindo_landmark_classifier.h5')
MODEL_BACKUP_V2  = os.path.join(BASE_DIR, 'models', 'bisindo_landmark_classifier_v2_backup.h5')
DICTIONARY_DIR   = os.path.join(BASE_DIR, 'dictionary')
RESULTS_DIR      = os.path.join(BASE_DIR, 'results')

# --- Hyperparameter ---
NUM_EPOCHS      = 200
BATCH_SIZE      = 64
LABEL_SMOOTHING = 0.1
AUGMENT_FACTOR  = 3    # Buat 3x jumlah data dengan augmentasi
NOISE_STD       = 0.01
SCALE_RANGE     = (0.85, 1.15)
TRANSLATE_RANGE = 0.05

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(DICTIONARY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# FUNGSI NORMALISASI LANDMARK
# =============================================================================

def normalize_landmarks(landmarks_flat):
    """
    Normalisasi landmark agar invariant terhadap posisi dan skala tangan.

    Metode: Wrist-relative + Scale normalization
    - Kurangi semua koordinat dengan posisi pergelangan tangan (wrist = landmark 0)
    - Bagi dengan jarak wrist ke pangkal jari tengah (landmark 9)
      sehingga ukuran tangan tidak mempengaruhi fitur

    Input : flat list/array panjang 126 (Left[63] + Right[63])
    Output: flat list panjang 126 yang sudah dinormalisasi
    """
    result = []
    for hand_start in [0, 63]:  # Left: index 0-62, Right: 63-125
        hand = np.array(landmarks_flat[hand_start:hand_start + 63],
                        dtype=np.float32).reshape(21, 3)

        # Jika tangan tidak terdeteksi (semua nol), biarkan nol
        if np.allclose(hand, 0):
            result.extend([0.0] * 63)
            continue

        # 1. Translasi: jadikan wrist (landmark 0) sebagai titik asal
        wrist = hand[0].copy()
        hand = hand - wrist

        # 2. Scaling: bagi dengan jarak wrist → middle finger MCP (landmark 9)
        #    Ini membuat fitur invariant terhadap ukuran/jarak tangan ke kamera
        scale = np.linalg.norm(hand[9])
        if scale > 1e-6:
            hand = hand / scale

        result.extend(hand.flatten().tolist())

    return result


def normalize_batch(X):
    """Normalisasi seluruh batch sekaligus."""
    return np.array([normalize_landmarks(row) for row in X], dtype=np.float32)


# =============================================================================
# FUNGSI AUGMENTASI DATA
# =============================================================================

def augment_landmarks(X, y, augment_factor=3,
                      noise_std=0.01,
                      scale_range=(0.85, 1.15),
                      translate_range=0.05):
    """
    Augmentasi data landmark dengan transformasi ringan.
    Menghasilkan data sintetis untuk meningkatkan generalisasi model.

    Transformasi:
    - Gaussian noise: variasi kecil koordinat (simulasi getaran tangan)
    - Random scale: variasi ukuran (simulasi jarak ke kamera)
    - Random translate: variasi posisi di frame
    """
    X_aug_list = [X]
    y_aug_list = [y]

    print(f"Augmentasi data: {augment_factor}x dari {len(X)} sampel...")
    for i in range(augment_factor):
        X_noisy = X.copy()

        for hand_start in [0, 63]:
            hand_data = X_noisy[:, hand_start:hand_start + 63]

            # Mask tangan yang ada data (tidak semua nol)
            has_hand = ~np.all(hand_data == 0, axis=1)

            if has_hand.any():
                # 1. Gaussian noise
                noise = np.random.normal(0, noise_std, hand_data[has_hand].shape)
                hand_data[has_hand] += noise

                # 2. Random scale (per sampel)
                scales = np.random.uniform(*scale_range, size=(has_hand.sum(), 1))
                hand_data[has_hand] *= scales

                # 3. Random translate (per sampel, per sumbu)
                translations = np.random.uniform(
                    -translate_range, translate_range,
                    size=(has_hand.sum(), 63)
                )
                hand_data[has_hand] += translations

            X_noisy[:, hand_start:hand_start + 63] = hand_data

        X_aug_list.append(X_noisy)
        y_aug_list.append(y)

    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.concatenate(y_aug_list)

    # Shuffle
    idx = np.random.permutation(len(X_augmented))
    return X_augmented[idx], y_augmented[idx]


# =============================================================================
# MAIN TRAINING
# =============================================================================

print("=" * 60)
print("  Training Model BISINDO v2 (Improved)")
print("=" * 60)

# --- 1. Load dan Pra-pemrosesan Data ---
print(f"\n[1/6] Memuat data dari: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}, Kelas: {df['label'].nunique()}")

X_raw = df.drop('label', axis=1).values.astype(np.float32)
y_labels = df['label'].values

# Encode label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

# Simpan nama kelas
class_names_path = os.path.join(DICTIONARY_DIR, 'class_names_landmark.txt')
with open(class_names_path, 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")
print(f"      Nama kelas disimpan di: {class_names_path}")
print(f"      Kelas: {list(class_names)}")

# --- 2. Normalisasi Fitur ---
print(f"\n[2/6] Normalisasi landmark (wrist-relative + scale-invariant)...")
X_normalized = normalize_batch(X_raw)
print(f"      Selesai. Shape: {X_normalized.shape}")

# --- 3. Split Data ---
X_train, X_val, y_train, y_val = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y_labels
)
print(f"\n[3/6] Split data:")
print(f"      Training  : {len(X_train)} sampel")
print(f"      Validasi  : {len(X_val)} sampel")

# --- 4. Augmentasi Data Training ---
print(f"\n[4/6] Augmentasi data training (factor={AUGMENT_FACTOR})...")
X_train_aug, y_train_aug = augment_landmarks(
    X_train, y_train,
    augment_factor=AUGMENT_FACTOR,
    noise_std=NOISE_STD,
    scale_range=SCALE_RANGE,
    translate_range=TRANSLATE_RANGE
)
print(f"      Data training setelah augmentasi: {len(X_train_aug)} sampel")

# One-hot encode untuk label smoothing
y_train_cat = to_categorical(y_train_aug, num_classes=num_classes)
y_val_cat   = to_categorical(y_val, num_classes=num_classes)

# --- 5. Bangun Model ---
print(f"\n[5/6] Membangun arsitektur model...")
model = Sequential([
    Input(shape=(X_train.shape[1],)),

    # Block 1
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Block 2
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    # Block 3
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Block 4
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    # Output
    Dense(num_classes, activation='softmax'),
], name='bisindo_mlp_v2')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=['accuracy']
)
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
]

# --- 6. Latih Model ---
print(f"\n[6/6] Memulai pelatihan ({NUM_EPOCHS} epoch maks, batch={BATCH_SIZE})...")
start_time = time.time()

history = model.fit(
    X_train_aug, y_train_cat,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks,
    verbose=1
)

duration = time.time() - start_time
print(f"\nPelatihan selesai dalam {duration:.1f} detik ({duration/60:.1f} menit)")

# Simpan juga sebagai backup v2
model.save(MODEL_BACKUP_V2)
print(f"Backup v2 disimpan di: {MODEL_BACKUP_V2}")

# =============================================================================
# EVALUASI
# =============================================================================
print("\n" + "=" * 60)
print("  EVALUASI MODEL")
print("=" * 60)

# Load model terbaik
model_best = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Prediksi pada data validasi
y_pred_proba = model_best.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification Report
print("\nClassification Report:")
report = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)
print(report)

# Simpan report ke file
report_path = os.path.join(RESULTS_DIR, 'classification_report_v2.txt')
with open(report_path, 'w') as f:
    f.write("Model v2 - Classification Report\n")
    f.write("=" * 50 + "\n")
    f.write(report)
print(f"Report disimpan: {report_path}")

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='lightgray', ax=ax)
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title('Confusion Matrix - Model BISINDO v2', fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_v2.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrix disimpan: {cm_path}")

# --- Grafik Training ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Akurasi
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='royalblue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='tomato')
axes[0].set_title('Akurasi Training vs Validasi', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Akurasi')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', color='royalblue')
axes[1].plot(history.history['val_loss'], label='Val Loss', color='tomato')
axes[1].set_title('Loss Training vs Validasi', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'training_history_v2.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Grafik training disimpan: {plot_path}")

# --- Ringkasan Akhir ---
val_loss, val_acc = model_best.evaluate(X_val, y_val_cat, verbose=0)
print("\n" + "=" * 60)
print(f"  HASIL AKHIR MODEL v2")
print("=" * 60)
print(f"  Val Accuracy : {val_acc*100:.2f}%")
print(f"  Val Loss     : {val_loss:.4f}")
print(f"  Model saved  : {MODEL_SAVE_PATH}")
print(f"  Backup v2    : {MODEL_BACKUP_V2}")
print(f"  Report       : {report_path}")
print(f"  Confusion Mat: {cm_path}")
print("=" * 60)
print("\nSelesai! Jalankan realtime_detector_mediapipe.py untuk mencoba model baru.")
