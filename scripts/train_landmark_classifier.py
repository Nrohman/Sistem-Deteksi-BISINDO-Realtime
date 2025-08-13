import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns # Untuk visualisasi confusion matrix
import time

# --- Konfigurasi Awal ---
DATA_PATH = 'data/bisindo_landmarks.csv'
MODEL_SAVE_PATH = 'models/bisindo_landmark_classifier.h5'
DICTIONARY_DIR = 'dictionary' # Folder untuk menyimpan class_names_landmark.txt
NUM_EPOCHS = 100 # Jumlah epoch pelatihan
BATCH_SIZE = 64 # Ukuran batch

# Pastikan direktori models dan dictionary ada
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(DICTIONARY_DIR, exist_ok=True)

# --- 1. Muat dan Pra-pemrosesan Data ---
print("Memuat data dari:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Pisahkan fitur (X) dan label (y)
X = df.drop('label', axis=1).values
y_labels = df['label'].values

# Encode label huruf menjadi angka
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_labels)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

# Simpan daftar nama kelas untuk digunakan nanti (misalnya di real-time detector)
class_names_path = os.path.join(DICTIONARY_DIR, 'class_names_landmark.txt')
with open(class_names_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"Nama kelas disimpan di: {class_names_path}")

# Bagi data menjadi set pelatihan dan validasi
# Stratify=y_labels memastikan distribusi kelas yang merata di train/val set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_labels)

print(f"Total data: {len(X)}")
print(f"Data pelatihan: {len(X_train)} sampel")
print(f"Data validasi: {len(X_val)} sampel")
print(f"Jumlah kelas: {num_classes}")

# --- 2. Bangun Model Multi-Layer Perceptron (MLP) ---
print("\nMembangun Model MLP...")
model = Sequential([
    # Input layer: jumlah neuron sesuai dimensi fitur (126 koordinat)
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3), # Mencegah overfitting
    Dense(128, activation='relu'),
    Dropout(0.3), # Mencegah overfitting
    # Output layer: jumlah neuron sesuai jumlah kelas, aktivasi softmax untuk klasifikasi multi-kelas
    Dense(num_classes, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Cocok untuk label integer
              metrics=['accuracy']) # Metrik utama yang akan dipantau

model.summary() # Menampilkan ringkasan arsitektur model

# --- 3. Konfigurasi Callbacks ---
# Early Stopping: Menghentikan pelatihan jika validasi loss tidak membaik
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model Checkpoint: Menyimpan model terbaik berdasarkan validasi akurasi
checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max', # Simpan model ketika val_accuracy maksimum
    verbose=1
)

# --- 4. Latih Model ---
# --- 4. Latih Model dengan Pengukuran Waktu ---
print(f"\nMemulai pelatihan model selama {NUM_EPOCHS} epoch...")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_callback, checkpoint_callback]
)

end_time = time.time()
training_duration = end_time - start_time
print(f"Pelatihan model selesai dalam {training_duration:.2f} detik.")

# Muat kembali model terbaik jika early stopping aktif
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
print(f"Model terbaik dimuat dari: {MODEL_SAVE_PATH}")


# --- 5. Evaluasi Model pada Data Validasi (Detail) ---
print("\n--- Evaluasi Model pada Data Validasi ---")

# Prediksi pada data validasi
y_pred_proba = model.predict(X_val)
y_pred = np.argmax(y_pred_proba, axis=1) # Ambil indeks kelas dengan probabilitas tertinggi

# Cetak Classification Report
# Ini akan menampilkan Precision, Recall, F1-score untuk setiap kelas dan rata-ratanya (macro avg, weighted avg)
print("\nClassification Report (Presisi, Recall, F1-score):")
print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0)) # zero_division=0 agar tidak warning/error jika ada kelas tidak terdeteksi

# Cetak Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_val, y_pred)
print(conf_matrix)

# Visualisasi Confusion Matrix (membutuhkan matplotlib dan seaborn)
plt.figure(figsize=(num_classes * 0.7, num_classes * 0.7)) # Ukuran figure disesuaikan dengan jumlah kelas
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=.5, linecolor='gray')
plt.xlabel('Prediksi Label')
plt.ylabel('Label Sebenarnya')
plt.title('Confusion Matrix Klasifikasi Huruf BISINDO')
plt.tight_layout()
plt.show()

# --- 6. Menampilkan Grafik Akurasi dan Loss ---
print("\nMenampilkan grafik Akurasi dan Loss...")
# Grafik Akurasi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Pelatihan')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Pelatihan')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Grafik Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nProses pelatihan dan evaluasi selesai.")