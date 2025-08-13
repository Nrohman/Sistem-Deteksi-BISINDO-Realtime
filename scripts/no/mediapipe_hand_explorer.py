import cv2
import mediapipe as mp
import time

def explore_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2, # <--- UBAH DI SINI DARI 1 MENJADI 2
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0) # 0 untuk webcam default
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return

    print("MediaPipe Hand Explorer dimulai. Mendeteksi hingga 2 tangan. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        # Balik frame secara horizontal untuk tampilan seperti cermin
        frame = cv2.flip(frame, 1)

        # Ubah BGR menjadi RGB karena MediaPipe mengharapkan input RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses gambar dan deteksi landmark tangan
        results = hands.process(image_rgb)

        # Gambar landmark pada frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar koneksi antara landmark
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Tampilkan frame
        cv2.imshow('MediaPipe Hand Explorer (2 Tangan)', frame)

        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close() # Penting: tutup objek hands MediaPipe

if __name__ == "__main__":
    explore_mediapipe_hands()