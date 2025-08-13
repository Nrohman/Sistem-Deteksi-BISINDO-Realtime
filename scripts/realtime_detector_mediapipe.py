import cv2
import numpy as np
import tensorflow as tf
import os
import time
import mediapipe as mp
from gtts import gTTS
import pygame
import threading

# --- Konfigurasi ---
MODEL_PATH = "models/bisindo_landmark_classifier.h5" # Model klasifikasi landmark baru
CLASS_NAMES_PATH = "dictionary/class_names_landmark.txt" # Nama kelas untuk model landmark
CONFIDENCE_THRESHOLD = 0.9 # Minimum confidence untuk menganggap deteksi valid (90%)
SPACE_DETECTION_THRESHOLD = 1.5 # Detik, jeda waktu untuk menganggap sebagai spasi antar huruf

# --- Inisialisasi Pygame Mixer untuk Audio ---
pygame.mixer.init()

# --- Inisialisasi MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, # Deteksi hingga 2 tangan
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Fungsi untuk Memuat Model dan Nama Kelas ---
def load_model_and_class_names(model_path, class_names_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model klasifikasi landmark berhasil dimuat dari: {model_path}")
    except Exception as e:
        print(f"Error memuat model klasifikasi landmark: {e}")
        return None, None

    try:
        with open(class_names_path, "r", encoding='utf-8') as f:
            class_names = [name.strip() for name in f.readlines()]
        print(f"Nama kelas untuk model landmark berhasil dimuat dari: {class_names_path}")
    except Exception as e:
        print(f"Error memuat nama kelas untuk model landmark: {e}")
        return None, None
    return model, class_names

# --- Fungsi Text-to-Speech ---
def text_to_speech(text, lang='id', slow=False):
    if not text:
        return

    # 1. Buat nama file audio yang unik menggunakan timestamp
    audio_filename = f"audio/temp_audio_{int(time.time())}.mp3"

    try:
        # 2. Hasilkan TTS dan simpan ke file unik
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(audio_filename)

        # 3. Fungsi untuk memutar dan menghapus audio di thread terpisah
        def play_and_clean_audio(filename):
            try:
                # Pastikan tidak ada musik lain yang sedang diputar (opsional, tapi baik)
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    time.sleep(0.05) # Beri sedikit jeda

                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                # Tunggu hingga musik selesai diputar
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Beri sedikit jeda ekstra untuk memastikan file dilepaskan oleh OS/Pygame
                time.sleep(0.1)

            except Exception as e:
                print(f"Error saat memutar audio {filename}: {e}")
                
            finally:
                # Hapus file audio setelah selesai diputar (atau jika ada error saat memutar)
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        # print(f"File {filename} berhasil dihapus.") # Debugging opsional
                    except OSError as e:
                        print(f"Peringatan: Gagal menghapus {filename}: {e}. Mungkin masih digunakan oleh OS.")
        
        # 4. Jalankan pemutaran dan pembersihan di thread terpisah
        audio_thread = threading.Thread(target=play_and_clean_audio, args=(audio_filename,))
        audio_thread.start()

    except Exception as e:
        print(f"Error saat text-to-speech untuk '{text}': {e}")
        print("Pastikan ada koneksi internet jika menggunakan gTTS (kecuali Anda menggunakan backend offline).")
# --- Fungsi untuk Mengekstrak dan Mempersiapkan Landmark ---
def extract_and_prepare_landmarks(results):
    """
    Mengekstrak landmark dari hasil MediaPipe dan mengaturnya ke format input model.
    Mengembalikan array numpy flat (126 fitur) atau None jika tidak ada tangan terdeteksi.
    """
    # Inisialisasi placeholder untuk 2 tangan (kiri dan kanan), masing-masing 21 landmark * 3 koordinat (x,y,z)
    hand_landmarks_dict = {
        'Left': [0.0] * (21 * 3), # Placeholder untuk tangan kiri
        'Right': [0.0] * (21 * 3) # Placeholder untuk tangan kanan
    }

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label
            
            current_hand_landmarks = []
            for landmark in hand_landmarks.landmark:
                current_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            if handedness in hand_landmarks_dict:
                hand_landmarks_dict[handedness] = current_hand_landmarks
            
    # Gabungkan landmark tangan kiri dan kanan dalam urutan yang konsisten
    final_landmarks = hand_landmarks_dict['Left'] + hand_landmarks_dict['Right']
    
    return np.array(final_landmarks).reshape(1, -1) # Reshape untuk input model (1, 126)


# --- Fungsi Utama Deteksi Real-time ---
def run_realtime_detection():
    model, class_names = load_model_and_class_names(MODEL_PATH, CLASS_NAMES_PATH)
    if model is None or class_names is None:
        print("Gagal memulai deteksi real-time karena model atau nama kelas tidak dapat dimuat.")
        hands_detector.close()
        pygame.mixer.quit()
        return

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        hands_detector.close()
        pygame.mixer.quit()
        return

    current_word_letters = [] # List untuk menyimpan huruf-huruf yang membentuk kata saat ini
    last_detection_time = time.time()
    displayed_text = "" # Teks yang akan ditampilkan di layar (hasil deteksi saat ini)
    final_spoken_word = "" # Kata terakhir yang diucapkan

    # Inisialisasi variabel untuk perhitungan FPS
    prev_frame_time = 0
    new_frame_time = 0

    # Pastikan folder audio ada
    os.makedirs("audio", exist_ok=True)

    print("Deteksi dimulai. Tunjukkan gestur BISINDO. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        frame = cv2.flip(frame, 1) # Balik frame secara horizontal
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses gambar dengan MediaPipe Hands
        results = hands_detector.process(image_rgb)

        predicted_letter = "Tidak Terdeteksi"
        confidence_score = 0.0
        
        # Gambar landmark pada frame
        if results.multi_hand_landmarks:
            # Ekstrak dan siapkan landmark untuk input model
            landmarks_input = extract_and_prepare_landmarks(results)
            
            # Prediksi huruf menggunakan model landmark
            predictions = model.predict(landmarks_input, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence_score = predictions[0][predicted_class_index]
            predicted_letter_raw = class_names[predicted_class_index]

            # Hanya gunakan prediksi jika confidence di atas ambang batas
            if confidence_score > CONFIDENCE_THRESHOLD:
                predicted_letter = predicted_letter_raw
            else:
                predicted_letter = "Tidak Yakin" # Atau bisa juga "Tidak Terdeteksi" jika confidence rendah

            # Gambar landmark di frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        current_time = time.time()

        # Logika deteksi spasi dan konversi kata
        if predicted_letter != "Tidak Terdeteksi" and predicted_letter != "Tidak Yakin":
            # Huruf terdeteksi dengan confidence tinggi
            if not current_word_letters or current_word_letters[-1] != predicted_letter:
                # Tambahkan huruf hanya jika berbeda dari huruf terakhir
                current_word_letters.append(predicted_letter)
                displayed_text = "".join(current_word_letters) # Update teks yang ditampilkan
            last_detection_time = current_time # Reset timer jeda

        elif (current_time - last_detection_time) > SPACE_DETECTION_THRESHOLD and current_word_letters:
            # Jeda terdeteksi DAN ada huruf yang terkumpul
            final_word = "".join(current_word_letters)
            
            # --- Auto-koreksi akan diintegrasikan di sini di Tahap 5.5 ---
            # Untuk saat ini, kita gunakan kata aslinya dulu
            # Misalnya: final_word_corrected = auto_correct_word(final_word)
            final_word_corrected = final_word # Placeholder untuk auto-koreksi

            if final_word_corrected and final_word_corrected != final_spoken_word: # Hanya ucapkan jika kata baru
                print(f"Kata terbentuk: {final_word_corrected}")
                text_to_speech(final_word_corrected)
                final_spoken_word = final_word_corrected
            current_word_letters = [] # Reset untuk kata berikutnya
            displayed_text = "" # Kosongkan tampilan setelah kata diucapkan

        # --- Bagian penambahan kode untuk FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        # --- End Bagian penambahan kode untuk FPS ---

        # Tampilkan informasi di frame (termasuk FPS)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # FPS di baris pertama
        display_letter_info = f"Deteksi: {predicted_letter} ({confidence_score:.2f})"
        cv2.putText(frame, display_letter_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # Deteksi di baris kedua

        display_word_info = f"Kata: {displayed_text}"
        cv2.putText(frame, display_word_info, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Kata di baris ketiga

        cv2.imshow('Deteksi Huruf BISINDO (MediaPipe)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close() # Tutup objek hands MediaPipe
    pygame.mixer.quit() # Tutup mixer Pygame

if __name__ == "__main__":
    # Pastikan folder models, dictionary, dan audio ada
    os.makedirs("audio", exist_ok=True) # Pastikan folder audio ada
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        print(f"Error: Folder models tidak ditemukan di {os.path.dirname(MODEL_PATH)}")
    elif not os.path.exists(CLASS_NAMES_PATH):
        print(f"Error: File class_names_landmark.txt tidak ditemukan di {CLASS_NAMES_PATH}")
        print("Pastikan scripts/train_landmark_classifier.py sudah dijalankan.")
    else:
        run_realtime_detection()