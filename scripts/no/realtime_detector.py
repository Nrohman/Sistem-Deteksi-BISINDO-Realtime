import cv2
import numpy as np
import tensorflow as tf
import os
import time
from gtts import gTTS
import pygame # Untuk memutar audio
import threading # Untuk memutar audio di background

# --- Konfigurasi ---
MODEL_PATH = "../models/bisindo_classifier_mobilenetv2.h5"
CLASS_NAMES_PATH = "../dictionary/class_names.txt" # Path ke file nama kelas
IMG_HEIGHT = 224
IMG_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence untuk menganggap deteksi valid
# Jeda waktu (detik) untuk menganggap sebagai spasi antar huruf
# Jika tidak ada deteksi valid selama periode ini, huruf sebelumnya dirangkai menjadi kata
SPACE_DETECTION_THRESHOLD = 1.5 # Detik

# --- Inisialisasi Pygame Mixer untuk Audio (diperlukan untuk gTTS) ---
pygame.mixer.init()

# --- Fungsi untuk Memuat Model dan Nama Kelas ---
def load_model_and_class_names(model_path, class_names_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model berhasil dimuat dari: {model_path}")
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None, None

    try:
        # Tambahkan encoding='utf-8' saat membuka file
        with open(class_names_path, "r", encoding='utf-8') as f:
            class_names = [name.strip() for name in f.readlines()]
        print(f"Nama kelas berhasil dimuat dari: {class_names_path}")
    except Exception as e:
        print(f"Error memuat nama kelas: {e}")
        return None, None
    return model, class_names

# --- Fungsi Text-to-Speech ---
def text_to_speech(text, lang='id', slow=False):
    if not text:
        return
    try:
        # Hapus file audio sebelumnya jika ada
        if os.path.exists("temp_audio.mp3"):
            os.remove("temp_audio.mp3")

        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save("temp_audio.mp3")

        # Memutar audio di thread terpisah agar tidak memblokir main loop
        def play_audio():
            pygame.mixer.music.load("temp_audio.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10) # Jaga agar tidak terlalu memakan CPU

        audio_thread = threading.Thread(target=play_audio)
        audio_thread.start()

    except Exception as e:
        print(f"Error saat text-to-speech: {e}")
        print("Pastikan ada koneksi internet jika menggunakan gTTS.")
        # Fallback ke pyttsx3 jika gTTS gagal (opsional, jika Anda ingin menambahkan)
        # import pyttsx3
        # engine = pyttsx3.init()
        # engine.say(text)
        # engine.runAndWait()


# --- Fungsi Utama Deteksi Real-time ---
def run_realtime_detection():
    model, class_names = load_model_and_class_names(MODEL_PATH, CLASS_NAMES_PATH)
    if model is None or class_names is None:
        print("Gagal memulai deteksi real-time karena model atau nama kelas tidak dapat dimuat.")
        return

    cap = cv2.VideoCapture(0) # 0 untuk webcam default
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    current_word = [] # List untuk menyimpan huruf-huruf yang membentuk kata saat ini
    last_detection_time = time.time() # Waktu deteksi huruf terakhir yang valid
    displayed_text = "" # Teks yang akan ditampilkan di layar
    last_spoken_word = "" # Kata terakhir yang diucapkan

    print("Deteksi dimulai. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        # Balik frame secara horizontal untuk tampilan seperti cermin
        frame = cv2.flip(frame, 1)

        # Pra-pemrosesan frame untuk input model
        img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        img = np.expand_dims(img, axis=0) # Tambahkan dimensi batch
        img = img / 255.0 # Normalisasi piksel

        # Prediksi huruf
        predictions = model.predict(img, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        predicted_letter = class_names[predicted_class_index]

        current_time = time.time()

        # Logika deteksi spasi dan konversi kata
        if confidence > CONFIDENCE_THRESHOLD:
            # Huruf terdeteksi dengan confidence tinggi
            if not current_word or current_word[-1] != predicted_letter:
                # Jika huruf baru atau berbeda dari huruf terakhir, tambahkan
                current_word.append(predicted_letter)
                displayed_text = "".join(current_word) # Update teks yang ditampilkan
            last_detection_time = current_time # Reset timer jeda

        elif (current_time - last_detection_time) > SPACE_DETECTION_THRESHOLD and current_word:
            # Jeda terdeteksi DAN ada huruf yang terkumpul
            final_word = "".join(current_word)
            if final_word != last_spoken_word: # Hanya ucapkan jika kata baru
                print(f"Kata terbentuk: {final_word}")
                text_to_speech(final_word)
                last_spoken_word = final_word
            current_word = [] # Reset untuk kata berikutnya
            displayed_text = "" # Kosongkan tampilan setelah kata diucapkan

        # Tampilkan huruf yang terdeteksi dan confidence
        display_text_on_frame = f"Deteksi: {predicted_letter} ({confidence:.2f})"
        cv2.putText(frame, display_text_on_frame, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Tampilkan kata yang sedang terbentuk
        cv2.putText(frame, f"Kata: {displayed_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        cv2.imshow('Deteksi Huruf BISINDO', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit() # Penting: matikan mixer Pygame saat aplikasi ditutup

if __name__ == "__main__":
    # Pastikan folder models dan dictionary ada
    # Tidak perlu mendefinisikan dictionary_folder di sini
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        print(f"Error: Folder models tidak ditemukan di {os.path.dirname(MODEL_PATH)}")
        print("Pastikan model sudah dilatih dan disimpan.")
    elif not os.path.exists(CLASS_NAMES_PATH):
        print(f"Error: File class_names.txt tidak ditemukan di {CLASS_NAMES_PATH}")
        print("Pastikan preprocessing.py sudah dijalankan.")
    else:
        run_realtime_detection()