import os
import sys

# Konfigurasi environment sebelum memuat library deep learning
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
os.environ.setdefault('KERAS_HOME', os.path.join(_BASE_DIR, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(_BASE_DIR, '.matplotlib'))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Tambahkan SCRIPT_DIR ke sys.path untuk import modul lokal
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

import cv2
import numpy as np
import tensorflow as tf
import time
import pickle
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from gtts import gTTS
import pygame
import threading
import urllib.request
from autocorrect import BisindoAutoCorrect

# --- Path relatif terhadap lokasi script ini ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# --- Konfigurasi File ---
MODEL_PATH             = os.path.join(BASE_DIR, "models", "bisindo_landmark_classifier.h5")
SCALER_PATH            = os.path.join(BASE_DIR, "models", "landmark_scaler.pkl")
CLASS_NAMES_PATH       = os.path.join(BASE_DIR, "dictionary", "class_names_landmark.txt")
DICTIONARY_WORDS_PATH  = os.path.join(BASE_DIR, "dictionary", "indonesian_dictionary.txt")
HAND_LANDMARKER_PATH   = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
AUDIO_DIR              = os.path.join(BASE_DIR, "audio")

# --- Parameter Deteksi & Stabilisasi ---
CONFIDENCE_THRESHOLD     = 0.75   # Minimum keyakinan model (75%)
STABLE_FRAMES_REQUIRED   = 6      # Jumlah frame berturut-turut untuk mengonfirmasi huruf (~0.2 detik)
MOTION_THRESHOLD         = 0.022  # Ambang batas pergerakan tangan (jika lebih = tangan sedang bergerak/transisi)

# --- Parameter Penyusunan Kata & Kalimat ---
HANDS_DOWN_TIMEOUT       = 2.0    # Detik tanpa tangan di kamera sebelum kata disuarakan
INACTIVITY_TIMEOUT       = 5.0    # Detik tanpa input huruf baru sebelum kata disuarakan

# Huruf BISINDO yang wajib menggunakan 2 tangan (berdasarkan dataset BISINDO)
TWO_HANDED_LETTERS = {'A', 'D', 'G', 'H', 'K', 'Q', 'S', 'T', 'W', 'X'}
ONE_HANDED_LETTERS = {'B', 'C', 'E', 'F', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'R', 'U', 'V', 'Y', 'Z'}

# Global scaler
_scaler = None

# URL model hand landmarker dari Google MediaPipe
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Koneksi tangan MediaPipe (21 landmark)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# --- Fungsi Download Model Hand Landmarker ---
def download_hand_landmarker():
    """Download model hand_landmarker.task jika belum ada."""
    if os.path.exists(HAND_LANDMARKER_PATH):
        return True
    print("Mendownload model MediaPipe Hand Landmarker (~29MB)...")
    try:
        os.makedirs(os.path.dirname(HAND_LANDMARKER_PATH), exist_ok=True)
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH)
        print("Download model hand landmarker selesai!")
        return True
    except Exception as e:
        print(f"Gagal mendownload model: {e}")
        return False


# --- Inisialisasi Audio ---
pygame.mixer.init()


# --- Fungsi Memuat Model, Label, dan Scaler ---
def load_model_and_class_names(model_path, class_names_path):
    global _scaler

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model klasifikasi dimuat dari: {model_path}")
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None, None

    try:
        with open(class_names_path, "r", encoding='utf-8') as f:
            class_names = [name.strip() for name in f.readlines() if name.strip()]
        print(f"Kelas ({len(class_names)}): {class_names}")
    except Exception as e:
        print(f"Error memuat nama kelas: {e}")
        return None, None

    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                _scaler = pickle.load(f)
            print(f"StandardScaler dimuat dari: {SCALER_PATH}")
        except Exception as e:
            print(f"Gagal memuat scaler ({e}), lanjut tanpa scaler.")
            _scaler = None
    else:
        print("StandardScaler tidak ditemukan.")

    return model, class_names


# --- Fungsi Text-to-Speech ---
def text_to_speech(text, lang='id', slow=False):
    if not text:
        return
    os.makedirs(AUDIO_DIR, exist_ok=True)
    audio_filename = os.path.join(AUDIO_DIR, f"temp_audio_{int(time.time()*1000)}.mp3")

    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(audio_filename)

        def play_and_clean_audio(filename):
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    time.sleep(0.05)
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error memutar audio: {e}")
            finally:
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except OSError:
                        pass

        audio_thread = threading.Thread(target=play_and_clean_audio, args=(audio_filename,))
        audio_thread.daemon = True
        audio_thread.start()
    except Exception as e:
        print(f"Error text-to-speech '{text}': {e}")


# --- Fungsi Ekstraksi Landmark dengan Penataan Spasial Layar ---
def extract_spatially_aligned_landmarks(detection_result):
    if not detection_result.hand_landmarks:
        return None, 0

    hands_data = []
    for hand_landmarks in detection_result.hand_landmarks:
        coords = []
        wrist_x = hand_landmarks[0].x
        for lm in hand_landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        hands_data.append({
            'wrist_x': wrist_x,
            'coords': coords,
            'raw_landmarks': hand_landmarks
        })

    num_hands = len(hands_data)

    if num_hands >= 2:
        # Urutkan secara spasial: tangan dengan x lebih kecil (kiri layar) jadi Left slot
        hands_data.sort(key=lambda h: h['wrist_x'])
        raw_landmarks = hands_data[0]['coords'] + hands_data[1]['coords']
        landmarks_array = np.array(raw_landmarks, dtype=np.float32).reshape(1, -1)
        if _scaler is not None:
            landmarks_array = _scaler.transform(landmarks_array).astype(np.float32)
        return [landmarks_array], num_hands

    elif num_hands == 1:
        # Jika hanya 1 tangan, buat 2 variasi penempatan (Left slot & Right slot)
        single_coords = hands_data[0]['coords']

        slot_left = np.zeros(126, dtype=np.float32)
        slot_left[:63] = single_coords

        slot_right = np.zeros(126, dtype=np.float32)
        slot_right[63:] = single_coords

        candidates = [slot_left.reshape(1, -1), slot_right.reshape(1, -1)]
        if _scaler is not None:
            candidates = [_scaler.transform(c).astype(np.float32) for c in candidates]

        return candidates, num_hands

    return None, 0


# --- Fungsi Mengukur Kecepatan Gerakan Landmark (Motion Filter) ---
def compute_hand_motion(prev_landmarks, current_landmarks):
    if prev_landmarks is None or current_landmarks is None:
        return 0.0

    total_dist = 0.0
    point_count = 0

    min_hands = min(len(prev_landmarks), len(current_landmarks))
    if min_hands == 0:
        return 0.0

    for h in range(min_hands):
        prev_h = prev_landmarks[h]
        curr_h = current_landmarks[h]
        key_indices = [0, 4, 8, 12, 16, 20]
        for idx in key_indices:
            dx = curr_h[idx].x - prev_h[idx].x
            dy = curr_h[idx].y - prev_h[idx].y
            dz = curr_h[idx].z - prev_h[idx].z
            total_dist += np.sqrt(dx*dx + dy*dy + dz*dz)
            point_count += 1

    return (total_dist / point_count) if point_count > 0 else 0.0


# --- Fungsi Menggambar Landmark di Frame ---
def draw_landmarks_on_frame(frame, detection_result):
    if not detection_result.hand_landmarks:
        return frame

    h, w, _ = frame.shape
    for hand_landmark_list in detection_result.hand_landmarks:
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmark_list[start_idx]
            end = hand_landmark_list[end_idx]
            start_pt = (int(start.x * w), int(start.y * h))
            end_pt = (int(end.x * w), int(end.y * h))
            cv2.line(frame, start_pt, end_pt, (0, 220, 0), 2, cv2.LINE_AA)

        for landmark in hand_landmark_list:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)

    return frame


# --- Fungsi Utama Deteksi Real-time ---
def run_realtime_detection():
    if not download_hand_landmarker():
        print("Tidak bisa melanjutkan tanpa model hand landmarker.")
        pygame.mixer.quit()
        return

    model, class_names = load_model_and_class_names(MODEL_PATH, CLASS_NAMES_PATH)
    if model is None or class_names is None:
        print("Gagal memulai deteksi: model atau kamus kelas tidak valid.")
        pygame.mixer.quit()
        return

    # Inisialisasi Modul Auto-Correct
    auto_corrector = BisindoAutoCorrect(DICTIONARY_WORDS_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam 0 tidak tersedia, mencoba index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Tidak dapat mengakses kamera.")
            landmarker.close()
            pygame.mixer.quit()
            return

    # --- Variabel Status Kalimat & Kata ---
    current_word_letters = []    # List huruf untuk kata aktif yang sedang dirangkai
    completed_sentences = []     # List kata yang sudah selesai dibentuk
    last_letter_added_time = time.time()
    last_hand_seen_time = time.time()
    prev_frame_time = 0
    timestamp_ms = 0

    # Temporal stability buffer
    candidate_letter = None
    candidate_count = 0
    cooldown_frames = 0
    flash_added_letter = ""
    flash_timer = 0

    # Auto-Correct Visual Banner
    correction_banner = ""
    correction_banner_timer = 0

    # Motion tracking
    prev_landmarks = None

    print("\n" + "=" * 65)
    print("  SISTEM PENYUSUN KATA BISINDO + AUTO-CORRECT REALTIME AKTIF")
    print("=" * 65)
    print("Petunjuk Penggunaan:")
    print("1. Tahan gestur stabil selama ~0.2 detik untuk mengetik huruf.")
    print("2. Kata TIDAK AKAN DIHAPUS selama tangan masih di depan kamera.")
    print("3. Sistem otomatis mengoreksi typo (misal: BISNIDO -> BISINDO).")
    print("4. Cara menyelesaikan kata:")
    print("   - Turunkan tangan selama 2 detik, ATAU")
    print("   - Tekan tombol [Space] di keyboard.")
    print("5. Kontrol Keyboard:")
    print("   - [Space]     : Selesaikan kata saat ini & bunyikan suara")
    print("   - [Backspace] : Hapus 1 huruf terakhir (bisa juga tekan 'b')")
    print("   - [c]         : Hapus kata yang sedang dirangkai")
    print("   - [r]         : Reset seluruh kalimat & kata")
    print("   - [q]         : Keluar dari aplikasi")
    print("=" * 65 + "\n")

    def finalize_and_speak_word():
        nonlocal current_word_letters, completed_sentences, last_letter_added_time
        nonlocal correction_banner, correction_banner_timer

        if not current_word_letters:
            return

        raw_word = "".join(current_word_letters)
        corrected_word, is_corrected, score = auto_corrector.correct(raw_word)

        if is_corrected:
            print(f">> [Auto-Correct] '{raw_word}' -> '{corrected_word}' ({int(score * 100)}%) | Mengucapkan audio...")
            correction_banner = f"Auto-Correct: {raw_word} -> {corrected_word} ({int(score * 100)}%)"
            correction_banner_timer = 70  # Munculkan notifikasi koreksi selama ~2.5 detik
        else:
            print(f">> Kata selesai: {corrected_word} | Mengucapkan audio...")
            correction_banner = ""
            correction_banner_timer = 0

        completed_sentences.append(corrected_word)
        text_to_speech(corrected_word)
        current_word_letters = []
        last_letter_added_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        frame = cv2.flip(frame, 1)  # Mirror horizontal
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms += 1
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_time = time.time()
        predicted_letter = "Tidak Terdeteksi"
        confidence_score = 0.0
        num_hands_detected = 0
        is_moving = False
        motion_val = 0.0

        if detection_result.hand_landmarks:
            num_hands_detected = len(detection_result.hand_landmarks)
            last_hand_seen_time = current_time

            # 1. Cek kecepatan gerak tangan (Motion Filter)
            motion_val = compute_hand_motion(prev_landmarks, detection_result.hand_landmarks)
            prev_landmarks = detection_result.hand_landmarks

            if motion_val > MOTION_THRESHOLD:
                is_moving = True
                predicted_letter = "Bergerak (Transisi)"
                candidate_letter = None
                candidate_count = 0
            else:
                # 2. Tangan relatif diam -> Ekstraksi spasial & prediksi
                candidate_inputs, hands_count = extract_spatially_aligned_landmarks(detection_result)

                if candidate_inputs is not None:
                    best_conf = -1.0
                    best_letter = None

                    for inp in candidate_inputs:
                        probs = model.predict(inp, verbose=0)[0]

                        # Aturan BISINDO: Jika hanya 1 tangan, mask huruf yang wajib 2 tangan
                        if hands_count == 1:
                            for idx, cname in enumerate(class_names):
                                if cname in TWO_HANDED_LETTERS:
                                    probs[idx] = 0.0

                        idx_max = np.argmax(probs)
                        conf = float(probs[idx_max])

                        if conf > best_conf:
                            best_conf = conf
                            best_letter = class_names[idx_max]

                    confidence_score = best_conf

                    if confidence_score >= CONFIDENCE_THRESHOLD and best_letter is not None:
                        predicted_letter = best_letter
                    else:
                        predicted_letter = f"Tidak Yakin ({best_letter})" if best_letter else "Tidak Yakin"

            # Gambar landmark di layar
            frame = draw_landmarks_on_frame(frame, detection_result)
        else:
            prev_landmarks = None
            candidate_letter = None
            candidate_count = 0

        # --- Logika Konfirmasi Huruf (Debounce) ---
        if cooldown_frames > 0:
            cooldown_frames -= 1

        if predicted_letter not in ("Tidak Terdeteksi", "Bergerak (Transisi)") and not predicted_letter.startswith("Tidak Yakin"):
            if predicted_letter == candidate_letter:
                candidate_count += 1
            else:
                candidate_letter = predicted_letter
                candidate_count = 1

            if candidate_count >= STABLE_FRAMES_REQUIRED and cooldown_frames == 0:
                current_word_letters.append(candidate_letter)
                last_letter_added_time = current_time
                flash_added_letter = candidate_letter
                flash_timer = 25
                cooldown_frames = 12
                candidate_count = 0
                print(f"[+] Huruf '{candidate_letter}' ditambahkan -> Kata: {''.join(current_word_letters)}")
        else:
            if not is_moving:
                candidate_count = max(0, candidate_count - 1)

        # --- Logika Penyelesaian Kata ---
        should_finalize = False
        remaining_countdown = None

        if len(current_word_letters) > 0:
            if num_hands_detected == 0:
                time_hands_down = current_time - last_hand_seen_time
                if time_hands_down >= HANDS_DOWN_TIMEOUT:
                    should_finalize = True
                else:
                    remaining_countdown = max(0.0, HANDS_DOWN_TIMEOUT - time_hands_down)
            else:
                time_idle = current_time - last_letter_added_time
                if time_idle >= INACTIVITY_TIMEOUT:
                    should_finalize = True

        if should_finalize:
            finalize_and_speak_word()
            remaining_countdown = None

        # --- Visualisasi Overlay HUD pada Layar ---
        h_frame, w_frame, _ = frame.shape
        overlay = frame.copy()
        hud_height = 165
        cv2.rectangle(overlay, (0, 0), (w_frame, hud_height), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Hitung FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Status Warna Deteksi
        if predicted_letter in class_names:
            status_color = (0, 255, 0)      # Hijau: Huruf Terdeteksi
        elif is_moving:
            status_color = (0, 215, 255)    # Oranye: Sedang Bergerak
        else:
            status_color = (180, 180, 180)  # Abu-abu: Belum Terdeteksi

        # Baris 1: Info Sistem, Tangan & Notifikasi Auto-Correct
        hands_str = f"{num_hands_detected} Tangan" if num_hands_detected > 0 else "Tangan Turun"
        cv2.putText(frame, f"BISINDO AI | FPS: {fps} | [{hands_str}]", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if correction_banner_timer > 0:
            correction_banner_timer -= 1
            cv2.putText(frame, f"[{correction_banner}]", (w_frame - 420, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        # Baris 2: Deteksi Huruf & Confidence
        conf_text = f"{int(confidence_score * 100)}%" if confidence_score > 0 else "-"
        cv2.putText(frame, f"Deteksi: {predicted_letter} ({conf_text})", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)

        # Progress bar kestabilan gestur
        progress_ratio = min(1.0, candidate_count / STABLE_FRAMES_REQUIRED)
        bar_x, bar_y, bar_w, bar_h = w_frame - 230, 42, 210, 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        if progress_ratio > 0:
            fill_w = int(bar_w * progress_ratio)
            fill_color = (0, 255, 0) if progress_ratio >= 1.0 else (0, 215, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
        cv2.putText(frame, "Stabil:", (bar_x - 65, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Feedback visual saat huruf baru berhasil ditambahkan (+A, +B, dll)
        if flash_timer > 0:
            flash_timer -= 1
            cv2.putText(frame, f"+{flash_added_letter}", (w_frame - 280, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Baris 3: Kata Aktif yang sedang dirangkai
        word_display = "".join(current_word_letters) if current_word_letters else "(Menunggu gestur...)"
        word_color = (0, 255, 255) if current_word_letters else (140, 140, 140)
        cv2.putText(frame, f"Kata Aktif : {word_display}", (15, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, word_color, 2, cv2.LINE_AA)

        # Countdown saat tangan turun
        if remaining_countdown is not None:
            cv2.putText(frame, f"[Selesai dlm {remaining_countdown:.1f}s]", (w_frame - 230, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

        # Baris 4: Riwayat Kalimat Lengkap
        sentence_display = " ".join(completed_sentences[-4:]) if completed_sentences else "-"
        cv2.putText(frame, f"Kalimat    : {sentence_display}", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Footer Bar: Petunjuk Navigasi Keyboard
        footer_y = h_frame - 10
        cv2.rectangle(frame, (0, h_frame - 35), (w_frame, h_frame), (20, 20, 20), -1)
        controls_text = "[Space] Selesai Kata | [B] Hapus Huruf | [C] Hapus Kata | [R] Reset | [Q] Keluar"
        cv2.putText(frame, controls_text, (15, footer_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow('Sistem Deteksi BISINDO Realtime', frame)

        # --- Penanganan Keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') or key == 13:  # Space atau Enter: Selesaikan kata manual
            finalize_and_speak_word()
        elif key == 8 or key == ord('b'):   # Backspace atau 'b': Hapus 1 huruf terakhir
            if current_word_letters:
                popped = current_word_letters.pop()
                print(f"[-] Menghapus huruf '{popped}' -> Kata: {''.join(current_word_letters)}")
                last_letter_added_time = current_time
        elif key == ord('c'):               # Clear kata aktif
            current_word_letters = []
            candidate_count = 0
            candidate_letter = None
            print("[x] Kata aktif dihapus.")
        elif key == ord('r'):               # Reset total (kalimat dan kata)
            current_word_letters = []
            completed_sentences = []
            candidate_count = 0
            candidate_letter = None
            correction_banner = ""
            print("[R] Seluruh kalimat dan kata direset.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    pygame.mixer.quit()
    print("Program selesai.")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Error: File model tidak ditemukan di {MODEL_PATH}")
    elif not os.path.exists(CLASS_NAMES_PATH):
        print(f"Error: File kamus kelas tidak ditemukan di {CLASS_NAMES_PATH}")
    else:
        run_realtime_detection()
