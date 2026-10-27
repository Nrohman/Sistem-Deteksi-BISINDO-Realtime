# Sistem Deteksi Bahasa Isyarat Indonesia (BISINDO) Realtime

Sistem penerjemah huruf **Bahasa Isyarat Indonesia (BISINDO)** secara *real-time* berbasis Computer Vision dan Machine Learning. Proyek ini mendeteksi gestur tangan melalui webcam menggunakan **MediaPipe Tasks Hand Landmarker**, mengklasifikasikannya dengan model **Neural Network (MLP)** berakurasi **98.76%**, merangkai huruf menjadi kata dengan **Auto-Correct Typo Bahasa Indonesia**, dan menyuarakannya secara otomatis melalui **Text-to-Speech (gTTS)**.

---

## 🌟 Fitur Utama

- **Deteksi Tangan & Landmark Real-time (MediaPipe Tasks API):** Menggunakan MediaPipe Tasks API terbaru yang kompatibel dengan Python modern (hingga Python 3.13), mendeteksi hingga 2 tangan dan mengekstrak 42 titik landmark 3D (126 fitur koordinat).
- **Klasifikasi Akurasi Tinggi (98.76% Akurasi Validasi):** Model Neural Network Multi-Layer Perceptron (MLP) yang dilatih menggunakan dataset bersih dengan normalisasi `StandardScaler`.
- **Spatial Alignment (Solusi Bug Mirror Kamera):** Penataan posisi tangan berbasis koordinat horizontal layar (Screen-X), memastikan gestur dua tangan (seperti huruf **A** dan **K**) tidak tertukar dengan huruf lain saat kamera di-mirror.
- **Motion Velocity Filter (Anti-Spam Gerakan Cepat):** Menghitung perpindahan landmark antar-frame; gerakan cepat atau membolak-balik tangan otomatis berstatus `Bergerak (Transisi)` dan tidak akan memicu huruf acak.
- **Temporal Debounce & Stabilization Bar:** Gestur harus ditahan stabil selama ~0.2 detik untuk dikonfirmasi. Terdapat indikator *progress bar* visual di layar.
- **Penyusunan Kata & Kalimat Alami:** Kata aktif tidak akan dihapus selama tangan masih berada di depan kamera. Kata diselesaikan saat tangan diturunkan selama 2 detik atau dengan menekan tombol `[Space]`.
- **Auto-Correct Typo Bahasa Indonesia Cerdas:** Otomatis memperbaiki huruf selip atau huruf ganda akibat menahan gestur (misalnya `BISNIDO` $\rightarrow$ `BISINDO`, `TRIMA` $\rightarrow$ `TERIMA`, `KASIHH` $\rightarrow$ `KASIH`) berbasis kamus kata bahasa Indonesia.
- **Text-to-Speech (gTTS) & Audio Player:** Kata yang telah selesai disusun otomatis diucapkan dengan intonasi bahasa Indonesia yang jelas menggunakan Google Text-to-Speech (gTTS) dan Pygame.
- **Head-Up Display (HUD) Interaktif:** Layar kamera menampilkan status deteksi huruf, persentase keyakinan, indikator kestabilan, kata aktif yang sedang dirangkai, riwayat kalimat, serta petunjuk gestur.

---

## 📊 Performa Model (v4 Clean)

Model dilatih ulang menggunakan dataset bersih setelah membuang 1.917 data baris kosong yang merusak model sebelumnya:

| Metrik Evaluasi | Model Asli (Sebelumnya) | Model v4 Clean (Sekarang) |
| :--- | :---: | :---: |
| **Akurasi Validasi** | ~75.9% | **98.76%** |
| **Validation Loss** | > 1.20 | **0.0668** |
| **Presisi Huruf A** | 16.0% (Sering tertukar Y) | **97.6% (100% Recall)** |
| **Presisi Huruf K** | ~79.0% (Tertukar P) | **100.0%** |
| **Presisi Huruf Y** | ~26.0% | **100.0%** |

Grafik kurva pelatihan dan matriks kebingungan (*confusion matrix*) lengkap tersimpan di folder [`results/`](results/).

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa:** Python 3.10 - 3.13
- **Computer Vision:** OpenCV (`opencv-python`), Google MediaPipe (`mediapipe >= 1.0` Tasks API)
- **Deep Learning / ML:** TensorFlow 2.x / Keras, Scikit-learn
- **Audio & TTS:** `gTTS` (Google Text-to-Speech), `pygame`
- **Manipulasi Data & Visualisasi:** `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 📁 Struktur Direktori

```text
Sistem-Deteksi-BISINDO-Realtime/
├── data/
│   ├── bisindo_Landmarks.csv          # Dataset koordinat landmark A-Z
│   └── readme_dataset.txt             # Sumber dataset Kaggle (Agung Ma'ruf)
├── dictionary/
│   ├── class_names_landmark.txt       # Daftar label kelas alfabet (A-Z)
│   └── indonesian_dictionary.txt      # Kamus kata bahasa Indonesia untuk Auto-Correct
├── models/
│   ├── bisindo_landmark_classifier.h5 # Model utama terlatih (98.76% akurasi)
│   ├── bisindo_landmark_classifier_ORIGINAL.h5 # Cadangan model asli versi awal
│   ├── landmark_scaler.pkl            # Objek StandardScaler untuk normalisasi fitur
│   └── hand_landmarker.task           # Model MediaPipe Tasks (didownload otomatis)
├── results/
│   ├── classification_report_clean.txt # Laporan metrik presisi/recall tiap kelas
│   ├── confusion_matrix_clean.png     # Visualisasi Confusion Matrix
│   └── training_history_clean.png     # Grafik akurasi & loss selama pelatihan
├── scripts/
│   ├── realtime_detector_mediapipe.py # Script utama deteksi webcam real-time
│   ├── autocorrect.py                 # Modul algoritma auto-correct typo
│   ├── train_model_clean.py           # Script pelatihan model dataset bersih (v4)
│   ├── diagnose_hands.py              # Script diagnosis & analisis koordinat landmark
│   └── extract_landmarks.py           # Script ekstraksi landmark dari gambar dataset
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/Nrohman/Sistem-Deteksi-BISINDO-Realtime.git
cd Sistem-Deteksi-BISINDO-Realtime
```

### 2. Buat & Aktifkan Virtual Environment
```bash
# Membuat virtual environment
python -m venv venv_apd

# Aktivasi di Windows (PowerShell):
.\venv_apd\Scripts\Activate.ps1

# Aktivasi di Linux / macOS:
source venv_apd/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```
*(Atau instal manual: `pip install tensorflow mediapipe opencv-python scikit-learn pandas gtts pygame seaborn matplotlib`)*

---

### 4. Menjalankan Deteksi Real-Time (Webcam)

Jalankan script utama:
```bash
python scripts/realtime_detector_mediapipe.py
```

*Saat pertama kali dijalankan, sistem akan otomatis mengunduh file model `hand_landmarker.task` (~29 MB) dari server resmi Google MediaPipe jika belum tersedia.*

---

## 🎮 Kontrol & Petunjuk Penggunaan

### Cara Mengetik Huruf & Kata:
1. **Membentuk Huruf:** Tunjukkan gestur alfabet BISINDO di depan kamera dan tahan stabil selama ~0.2 detik hingga indikator bar hijau penuh.
2. **Huruf Dua Tangan:** Di BISINDO, huruf **A, D, G, H, K, Q, S, T, W, X** wajib menggunakan **2 TANGAN**. Pastikan kedua tangan terlihat di kamera.
3. **Penyusunan Kata:** Kata yang sedang dirangkai tidak akan hilang selama tangan Anda masih berada di kamera.
4. **Menyelesaikan Kata & Membunyikan Suara:**
   - **Otomatis:** Turunkan tangan dari pandangan kamera selama 2 detik.
   - **Manual:** Tekan tombol **`[Space]`** di keyboard untuk langsung menyelesaikan kata saat itu juga.

### Tombol Navigasi Keyboard:
| Tombol | Fungsi |
| :---: | :--- |
| **`[Space]` / `[Enter]`** | Selesaikan kata saat ini, lakukan auto-correct, dan bunyikan suara (TTS) |
| **`[Backspace]` / `'b'`** | Hapus 1 huruf terakhir dari kata yang sedang dirangkai |
| **`'c'`** | Hapus (*clear*) seluruh huruf pada kata yang sedang dirangkai |
| **`'r'`** | Reset total (*clear* kata aktif dan seluruh riwayat kalimat) |
| **`'q'`** | Keluar dari aplikasi deteksi |

---

## 🔄 Melatih Ulang Model (Opsional)

Jika Anda ingin melatih ulang model menggunakan dataset landmark:
```bash
python scripts/train_model_clean.py
```
Script ini akan:
1. Membaca `data/bisindo_Landmarks.csv` dan membersihkan baris kosong (semua 0).
2. Menerapkan `StandardScaler` dan menyimpan scaler ke `models/landmark_scaler.pkl`.
3. Melatih MLP dengan arsitektur BatchNormalization & Dropout selama hingga 120 epoch.
4. Menyimpan model terbaik ke `models/bisindo_landmark_classifier.h5`.
5. Menghasilkan laporan evaluasi dan grafik ke folder `results/`.

---

## 👥 Kontributor
* **Nurohman**
* Proyek Tugas UAS Machine Learning Lanjutan - Universitas Krisnadwipayana
