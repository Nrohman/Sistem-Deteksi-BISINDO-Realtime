import os
import sys

# Setup direktori cache agar tidak ada permission issue di sandbox
os.environ['KERAS_HOME'] = os.path.abspath('.keras')
os.environ['MPLCONFIGDIR'] = os.path.abspath('.matplotlib')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Path Konfigurasi ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(BASE_DIR, 'data', 'bisindo_Landmarks.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'bisindo_landmark_classifier.h5')
BACKUP_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'bisindo_landmark_classifier_v4_clean.h5')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'landmark_scaler.pkl')
DICTIONARY_DIR = os.path.join(BASE_DIR, 'dictionary')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(DICTIONARY_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("  PELATIHAN MODEL BISINDO (CLEAN DATASET & STANDARDIZATION)")
print("=" * 65)

# --- 1. Muat dan Bersihkan Data ---
print(f"\n[1/6] Memuat data dari: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
initial_count = len(df)
print(f"      Total sampel awal: {initial_count}")

# PEMBERSIHAN DATA: Hapus baris yang seluruhnya bernilai 0
# (Gambar yang gagal dideteksi tangannya oleh MediaPipe saat ekstraksi)
features = df.drop('label', axis=1).values
valid_mask = (np.abs(features).sum(axis=1) > 0)
df_clean = df[valid_mask].copy()
removed_count = initial_count - len(df_clean)

print(f"      Dihapus baris kosong (semua 0): {removed_count} baris")
print(f"      Total sampel bersih yang valid: {len(df_clean)} baris")

X = df_clean.drop('label', axis=1).values.astype(np.float32)
y_labels = df_clean['label'].values

# --- 2. Encode Label ---
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

class_names_path = os.path.join(DICTIONARY_DIR, 'class_names_landmark.txt')
with open(class_names_path, 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(f"{name}\n")
print(f"      {num_classes} kelas: {', '.join(class_names)}")
print(f"      Nama kelas disimpan di: {class_names_path}")

# --- 3. Split Dataset ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[2/6] Split data: Train = {len(X_train)} sampel, Val = {len(X_val)} sampel")

# --- 4. Standard Scaler ---
print(f"\n[3/6] Standarisasi fitur menggunakan StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)

with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"      Scaler disimpan di: {SCALER_SAVE_PATH}")

# Ringan augmentasi dengan sedikit Gaussian noise pada data latih (hanya titik aktif)
print(f"\n[4/6] Menambahkan augmentasi variasi posisi...")
noise_std = 0.005
noise = np.random.normal(0, noise_std, X_train_scaled.shape).astype(np.float32)
mask = (X_train != 0).astype(np.float32)
X_train_noisy = X_train_scaled + noise * mask

X_train_final = np.vstack([X_train_scaled, X_train_noisy])
y_train_final = np.concatenate([y_train, y_train])
shuffle_idx = np.random.permutation(len(X_train_final))
X_train_final = X_train_final[shuffle_idx]
y_train_final = y_train_final[shuffle_idx]
print(f"      Total data latih setelah augmentasi: {len(X_train_final)} sampel")

# --- 5. Bangun Arsitektur Model ---
print(f"\n[5/6] Membangun arsitektur Neural Network MLP...")
model = Sequential([
    Input(shape=(126,)),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
], name='bisindo_classifier_clean')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

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
        min_lr=1e-5,
        verbose=1
    )
]

# --- 6. Pelatihan ---
print(f"\n[6/6] Memulai pelatihan model...")
start_time = time.time()
history = model.fit(
    X_train_final, y_train_final,
    epochs=120,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=callbacks,
    verbose=1
)
elapsed = time.time() - start_time
print(f"\nPelatihan selesai dalam {elapsed:.1f} detik ({elapsed/60:.1f} menit)")

# Simpan juga cadangan v4
model.save(BACKUP_SAVE_PATH)
print(f"Cadangan model disimpan di: {BACKUP_SAVE_PATH}")

# --- Evaluasi ---
print("\n" + "=" * 65)
print("  EVALUASI PERFORMA MODEL")
print("=" * 65)

best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
y_pred_probs = best_model.predict(X_val_scaled, verbose=0)
y_pred = y_pred_probs.argmax(axis=1)

report = classification_report(y_val, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:")
print(report)

# Simpan classification report
report_path = os.path.join(RESULTS_DIR, 'classification_report_clean.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Classification Report - Model BISINDO Clean\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
print(f"Laporan evaluasi disimpan di: {report_path}")

# Confusion Matrix
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='lightgray')
plt.title('Confusion Matrix - Model BISINDO Clean', fontsize=14, fontweight='bold')
plt.xlabel('Prediksi', fontsize=12)
plt.ylabel('Aktual', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_clean.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix disimpan di: {cm_path}")

# Training History Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Latih', color='royalblue')
plt.plot(history.history['val_accuracy'], label='Validasi', color='tomato')
plt.title('Akurasi Model', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Latih', color='royalblue')
plt.plot(history.history['val_loss'], label='Validasi', color='tomato')
plt.title('Loss Model', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

history_path = os.path.join(RESULTS_DIR, 'training_history_clean.png')
plt.savefig(history_path, dpi=150)
plt.close()
print(f"Grafik pelatihan disimpan di: {history_path}")

val_loss, val_acc = best_model.evaluate(X_val_scaled, y_val, verbose=0)
print("\n" + "=" * 65)
print(f"  HASIL AKHIR PELATIHAN")
print("=" * 65)
print(f"  Akurasi Validasi : {val_acc * 100:.2f}%")
print(f"  Loss Validasi    : {val_loss:.4f}")
print(f"  Model Utama      : {MODEL_SAVE_PATH}")
print(f"  Model Backup     : {BACKUP_SAVE_PATH}")
print(f"  Scaler           : {SCALER_SAVE_PATH}")
print("=" * 65)
