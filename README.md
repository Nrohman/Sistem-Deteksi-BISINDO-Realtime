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
* **Python 3.x**
* **OpenCV:** Untuk pemrosesan gambar dan tampilan video.
* **MediaPipe:** Untuk deteksi tangan dan ekstraksi landmark.
* **TensorFlow / Keras:** Untuk membangun dan melatih model Multi-Layer Perceptron (MLP).
* **gTTS (Google Text-to-Speech):** Untuk konversi teks ke suara.
* **Pygame (atau Playsound):** Untuk pemutaran audio.
* `numpy`
* `pandas` (mungkin diperlukan untuk pra-pemrosesan data)

## Struktur Proyek
Sistem-Deteksi-BISINDO-Realtime/
├── realtime_detector_bisindo.py # Script utama untuk sistem deteksi real-time
├── extract_landmarks.py         # Script untuk ekstraksi landmark dari dataset
├── train_landmark_classifier.py # Script untuk melatih model MLP
├── data/                        # Folder untuk dataset gambar mentah BISINDO
│   └── alphabet/                # Subfolder contoh: a, b, c, ...
│       └── ...
├── data/bisindo_landmarks.csv   # Hasil ekstraksi landmark (seringkali besar, bisa diabaikan di .gitignore)
├── dictionary/
│   └── class_names_landmark.txt # File berisi nama-nama kelas huruf
└── models/
└── bisindo_landmark_classifier.h5 # Model MLP yang sudah terlatih

## Cara Menjalankan
### Prasyarat:
1.  **Kloning repositori ini:**
    ```bash
    git clone [https://github.com/NamaPenggunaAnda/Sistem-Deteksi-BISINDO-Realtime.git](https://github.com/NamaPenggunaAnda/Sistem-Deteksi-BISINDO-Realtime.git)
    cd Sistem-Deteksi-BISINDO-Realtime
    ```
2.  **Instal dependensi:**
    ```bash
    pip install opencv-python mediapipe tensorflow gtts playsound pandas # playsound untuk lokal
    ```
    *Catatan: Jika Anda berencana melatih ulang model, Anda mungkin memerlukan dataset gambar BISINDO yang lebih besar di folder `data/alphabet/`.*

### Eksekusi:
1.  **Ekstraksi Landmark (jika Anda memiliki dataset gambar baru atau ingin melatih ulang):**
    ```bash
    python extract_landmarks.py
    ```
2.  **Pelatihan Model (jika Anda ingin melatih ulang model):**
    ```bash
    python train_landmark_classifier.py
    ```
3.  **Jalankan Sistem Deteksi Real-time:**
    ```bash
    python realtime_detector_bisindo.py
    ```

## Kontributor
Nurohman
**Tugas UAS Machine Learning Lanjutan - Universitas Krisnadwipayana / Machine Learning**

