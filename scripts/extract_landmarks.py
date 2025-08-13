import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm # Untuk menampilkan progress bar

# --- Konfigurasi ---
DATA_DIRECTORY = "data/alfabet" # <--- UBAH BARIS INI
OUTPUT_CSV_PATH = "data/bisindo_landmarks.csv"

# --- Inisialisasi MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, # Set ke True karena kita memproses gambar statis
    max_num_hands=2,       # Deteksi hingga 2 tangan
    min_detection_confidence=0.5
)

def extract_landmarks_from_image(image_path):
    """
    Mengekstrak landmark tangan dari sebuah gambar.
    Mengembalikan list landmark (x, y, z) untuk setiap tangan.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Peringatan: Tidak dapat memuat gambar {image_path}")
        return None

    # Ubah BGR menjadi RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses gambar
    results = hands.process(image_rgb)

    landmarks_data = []
    # MediaPipe memberikan multi_hand_landmarks dan multi_handedness
    # multi_handedness akan memberitahu apakah itu tangan kiri/kanan
    
    # Inisialisasi placeholder untuk 2 tangan (kiri dan kanan), masing-masing 21 landmark * 3 koordinat (x,y,z)
    # Total 126 fitur (2 tangan * 21 landmark * 3 koordinat)
    # Kita akan flattern nanti menjadi 1D array
    hand_landmarks_dict = {
        'Left': [0.0] * (21 * 3), # Placeholder untuk tangan kiri
        'Right': [0.0] * (21 * 3) # Placeholder untuk tangan kanan
    }

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label # 'Left' atau 'Right'
            
            # Ubah landmark menjadi list flat [x0, y0, z0, x1, y1, z1, ...]
            current_hand_landmarks = []
            for landmark in hand_landmarks.landmark:
                # Normalisasi koordinat: x,y,z relatif terhadap lebar/tinggi gambar
                # MediaPipe memberikan koordinat relatif 0-1, jadi kita bisa langsung menggunakannya.
                current_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Simpan berdasarkan handedness
            if handedness in hand_landmarks_dict:
                hand_landmarks_dict[handedness] = current_hand_landmarks
            
    # Gabungkan landmark tangan kiri dan kanan dalam urutan yang konsisten
    # Contoh: [L_x0, L_y0, L_z0, ..., L_x20, L_y20, L_z20, R_x0, R_y0, R_z0, ..., R_x20, R_y20, R_z20]
    final_landmarks = hand_landmarks_dict['Left'] + hand_landmarks_dict['Right']
    
    return final_landmarks


def process_dataset_and_extract_landmarks(data_dir, output_csv_path):
    """
    Memproses seluruh dataset gambar, mengekstrak landmark, dan menyimpannya ke CSV.
    """
    all_landmarks_data = []
    labels = []
    
    # Pastikan folder output ada
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Dapatkan nama-nama kelas (huruf) dari folder dataset
    class_names = sorted(os.listdir(data_dir))
    class_names = [c for c in class_names if os.path.isdir(os.path.join(data_dir, c))]
    
    print(f"Mulai mengekstrak landmark dari {len(class_names)} kelas.")

    for class_name in tqdm(class_names, desc="Memproses Kelas"):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            
            landmarks = extract_landmarks_from_image(image_path)
            
            if landmarks is not None and len(landmarks) == (2 * 21 * 3): # Pastikan 42 landmark * 3 koordinat
                all_landmarks_data.append(landmarks)
                labels.append(class_name)
            # else:
            #     print(f"Melewatkan gambar {image_path} karena tidak ada landmark atau jumlahnya tidak sesuai.")
                
    # Buat DataFrame Pandas
    # Nama kolom akan menjadi 'x0_L', 'y0_L', 'z0_L', ..., 'x20_R', 'y20_R', 'z20_R', 'label'
    columns = []
    for hand_type in ['Left', 'Right']:
        for i in range(21):
            columns.extend([f'x{i}_{hand_type}', f'y{i}_{hand_type}', f'z{i}_{hand_type}'])
    columns.append('label')

    df = pd.DataFrame(all_landmarks_data, columns=columns[:-1])
    df['label'] = labels

    df.to_csv(output_csv_path, index=False)
    print(f"\nEkstraksi landmark selesai. Data disimpan ke {output_csv_path}")
    print(f"Jumlah sampel yang diekstrak: {len(df)}")

if __name__ == "__main__":
    print("Pastikan Anda telah menginstal `tqdm` (`pip install tqdm`) dan `pandas` (`pip install pandas`)")
    # Pastikan MediaPipe hands object ditutup
    try:
        process_dataset_and_extract_landmarks(DATA_DIRECTORY, OUTPUT_CSV_PATH)
    finally:
        hands.close()