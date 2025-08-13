# Sistem-Deteksi-BISINDO-Realtime
Sistem deteksi huruf Bahasa Isyarat Indonesia (BISINDO) real-time menggunakan MediaPipe dan Multi-Layer Perceptron (MLP) untuk konversi isyarat tangan ke teks dan suara

# Sistem Deteksi Bahasa Isyarat Indonesia (BISINDO) Real-time
Proyek ini mengimplementasikan sistem deteksi huruf Bahasa Isyarat Indonesia (BISINDO) secara real-time menggunakan kombinasi Computer Vision (MediaPipe Hands) dan Machine Learning (Multi-Layer Perceptron/MLP). Tujuannya adalah untuk mengkonversi isyarat tangan BISINDO menjadi teks dan suara, memfasilitasi komunikasi yang lebih inklusif.


## Fitur Utama
* **Deteksi Tangan dan Ekstraksi Landmark Real-time:** Memanfaatkan MediaPipe Hands untuk secara akurat mendeteksi tangan dan mengekstrak 21 landmark 3D dari setiap tangan.
* **Klasifikasi Huruf BISINDO:** Menggunakan model Multi-Layer Perceptron (MLP) yang terlatih untuk mengklasifikasikan pola landmark tangan menjadi huruf-huruf alfabet BISINDO.
* **Pembentukan Kata:** Mengakumulasikan huruf-huruf yang diprediksi untuk membentuk kata, dengan logika deteksi spasi berdasarkan jeda waktu isyarat.
* **Konversi Teks ke Suara (Text-to-Speech):** Mengintegrasikan modul konversi teks ke suara (gTTS) untuk mengubah kata-kata yang terbentuk menjadi output suara yang jelas.
* **Pemutaran Audio:** Menggunakan Pygame (atau modul lain seperti Playsound di lingkungan lokal) untuk memutar output suara, memungkinkan komunikasi dua arah.
* **Tampilan Visual Real-time:** Menampilkan *feed* kamera dengan visualisasi deteksi tangan, prediksi huruf, dan teks yang terbentuk secara langsung.


## Tahapan Pengembangan
Proyek ini dikembangkan melalui tahapan-tahapan berikut:
1.  **Pengumpulan Data:** Memanfaatkan dataset gambar huruf BISINDO.
2.  **Pra-pemrosesan Data & Ekstraksi Fitur Landmark:** Gambar diproses oleh MediaPipe Hands untuk mengekstrak 126 fitur landmark per isyarat, yang disimpan dalam format CSV.
3.  **Pelatihan Model Multi-Layer Perceptron (MLP):** Data landmark dilatih menggunakan model MLP. Model terlatih (`.h5`) dan nama-nama kelas disimpan untuk digunakan dalam deteksi real-time.
4.  **Implementasi Sistem Real-time:** Aplikasi membaca input dari webcam, mengekstrak landmark, memprediksi huruf dengan model MLP, dan menampilkan hasilnya.
5.  **Integrasi Konversi Teks dan Suara:** Teks prediksi diakumulasikan menjadi kata, deteksi spasi dilakukan berdasarkan jeda waktu, dan kata dikonversi menjadi suara yang kemudian diputar.
6.  **Pengujian dan Evaluasi:** Evaluasi kinerja sistem secara keseluruhan, termasuk akurasi deteksi huruf, pembentukan kata, dan respons suara.


## Teknologi yang Digunakan
* **Python:** Versi 3.10.10
* **pandas:** Untuk manipulasi dan analisis data (khususnya untuk file CSV landmark).
* **tqdm:** Untuk menampilkan progress bar saat memproses data atau melatih model.
* **opencv-python (OpenCV):** Untuk pemrosesan gambar dan tampilan video.
* **mediapipe:** Untuk deteksi tangan dan ekstraksi landmark.
* **scikit-learn:** Untuk tugas-tugas Machine Learning seperti pra-pemrosesan data, pembagian dataset, atau evaluasi metrik (jika digunakan).
* **tensorflow:** Untuk membangun dan melatih model Multi-Layer Perceptron (MLP).
* **seaborn:** Untuk visualisasi data, misalnya untuk menampilkan confusion matrix atau grafik kinerja model.
* **gtts (Google Text-to-Speech):** Untuk konversi teks ke suara.
* **pygame:** Untuk pemutaran audio.

## Struktur Proyek
DeteksiHuruf/

├── audio/                          # Folder untuk file audio (jika ada, misal hasil gTTS sementara)

├── data/

│   ├── alfabet/                    # Subfolder untuk dataset gambar per huruf (A, B, C, ...)

│   │   ├── A/

│   │   ├── B/

│   │   └── ...

│   └── bisindo_landmarks.csv       # File CSV hasil ekstraksi landmark dari dataset

├── dictionary/

│   └── class_names_landmark.txt    # File berisi nama-nama kelas huruf yang dideteksi

├── models/

│   └── bisindo_landmark_classifier.h5 # Model MLP yang sudah terlatih

├── scripts/

│   └── extract_landmarks.py            # Script untuk ekstraksi landmark dari dataset gambar

│   └── realtime_detector_bisindo.py    # Script utama untuk sistem deteksi real-time

│   └── train_landmark_classifier.py    # Script untuk melatih model MLP

└── venv_apd/                       # Folder untuk virtual environment Python

*(Catatan data: data pada Folder alfabet A-Z dapat diunduh sesuai keterangan pada readme_dataset.txt)*
*(Catatan: Folder `audio/` dan `no/` ditambahkan sesuai gambar struktur direktori yang diberikan. Sesuaikan jika penggunaannya berbeda.)*

## Cara Menjalankan

### Prasyarat:

Kloning repositori Anda:
```Buka terminal atau Git Bash, lalu ketik perintah ini. Pastikan Anda mengganti NamaPenggunaAnda dengan username GitHub Anda.```

Bash
```
git clone https://github.com/Nrohman/Sistem-Deteksi-BISINDO-Realtime.git
cd Sistem-Deteksi-BISINDO-Realtime
```
Buat dan aktifkan virtual environment (sangat direkomendasikan):
Ini akan membantu mengelola dependensi proyek Anda secara terisolasi.


Bash
```
python -m venv venv_apd
# Untuk Windows:
.\venv_apd\Scripts\activate
# Untuk macOS/Linux:
source venv_apd/bin/activate
```
Instal semua dependensi:
Pastikan Anda memiliki koneksi internet aktif saat menjalankan perintah ini karena akan mengunduh semua library yang diperlukan.


Bash
```
pip install pandas tqdm opencv-python mediapipe scikit-learn tensorflow seaborn gtts pygame
(Catatan: Jika Anda berencana melatih ulang model atau menggunakan fitur yang memerlukan dataset, Anda mungkin memerlukan dataset gambar BISINDO yang lebih besar di folder data/alphabet/.)
```
Eksekusi Proyek:
Setelah prasyarat terpenuhi, Anda dapat menjalankan script proyek Anda sesuai tahapan yang Anda sebutkan:
Ekstraksi Landmark (jika Anda memiliki dataset gambar baru atau ingin melatih ulang):

Bash
```
python extract_landmarks.py
```
Pelatihan Model (jika Anda ingin melatih ulang model):
Bash
```
python train_landmark_classifier.py
```
Jalankan Sistem Deteksi Real-time:
Ini adalah script utama yang akan memulai deteksi BISINDO secara real-time menggunakan webcam Anda.

Bash
```
python realtime_detector_bisindo.py
```
Pastikan Anda menjalankan perintah ini di terminal yang sama di mana virtual environment venv_apd Anda sudah aktif.


## Kontributor
Nurohman
**Tugas UAS Machine Learning Lanjutan - Universitas Krisnadwipayana / Machine Learning**

