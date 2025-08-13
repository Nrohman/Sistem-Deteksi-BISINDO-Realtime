import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import os
import datetime # Untuk nama log TensorBoard

# Import fungsi pra-pemrosesan dari script sebelumnya
from preprocessing import load_and_preprocess_image_dataset

def train_classifier_model(data_dir, img_height, img_width, batch_size, epochs, model_save_path):
    """
    Membangun, melatih, dan menyimpan model klasifikasi huruf BISINDO menggunakan MobileNetV2.
    """
    # 1. Muat Dataset
    print("Memuat dataset pelatihan dan validasi untuk model klasifikasi...")
    train_ds, class_names = load_and_preprocess_image_dataset(
        data_dir, img_height, img_width, batch_size,
        validation_split=0.2, subset='training'
    )
    val_ds, _ = load_and_preprocess_image_dataset( # Tidak perlu class_names lagi dari val_ds
        data_dir, img_height, img_width, batch_size,
        validation_split=0.2, subset='validation'
    )

    num_classes = len(class_names)
    print(f"Jumlah kelas yang akan diklasifikasikan: {num_classes}")
    print(f"Nama kelas: {class_names}")

    # Konfigurasi dataset untuk performa (prefetching & caching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Membangun Model (MobileNetV2)
    # Gunakan MobileNetV2 sebagai base model tanpa bagian klasifikasi teratas (include_top=False)
    # Ini akan memuat bobot yang sudah dilatih di ImageNet
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False, # Jangan sertakan lapisan klasifikasi ImageNet
        weights='imagenet' # Gunakan bobot pre-trained dari ImageNet
    )

    # Bekukan base model agar bobotnya tidak berubah selama pelatihan awal
    base_model.trainable = False

    # Bangun model Sequential
    model = Sequential([
        base_model, # Lapisan base MobileNetV2
        GlobalAveragePooling2D(), # Mengubah output MobileNetV2 menjadi vektor tunggal
        Dense(num_classes, activation='softmax') # Lapisan output untuk klasifikasi ke jumlah kelas Anda
    ])

    # 3. Kompilasi Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), # Optimizer Adam
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Loss function untuk klasifikasi multi-kelas
        metrics=['accuracy'] # Metrik yang akan dipantau
    )

    model.summary()

    # 4. Callback untuk menyimpan log pelatihan (opsional tapi disarankan)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 5. Latih Model
    print(f"\nMemulai pelatihan model selama {epochs} epoch...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[tensorboard_callback]
    )

    # 6. Evaluasi Model (opsional)
    print("\nEvaluasi model pada dataset validasi:")
    loss, accuracy = model.evaluate(val_ds)
    print(f"Loss validasi: {loss:.4f}")
    print(f"Akurasi validasi: {accuracy:.4f}")

    # 7. Simpan Model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Pastikan folder model ada
    model.save(model_save_path)
    print(f"\nModel berhasil disimpan di: {model_save_path}")

if __name__ == "__main__":
    DATA_DIRECTORY = "data"
    IMG_HEIGHT = 224 # MobileNetV2 direkomendasikan pada 224x224
    IMG_WIDTH = 224
    BATCH_SIZE = 32 # Jangan terlalu besar agar tidak memakan RAM berlebih
    EPOCHS = 15 # Anda bisa menyesuaikan jumlah epoch ini. Mulai dengan 10-15.
                # Jika akurasi masih meningkat, Anda bisa menambahkannya.
    MODEL_SAVE_PATH = "../models/bisindo_classifier_mobilenetv2.h5" # Path untuk menyimpan model

    # Pastikan direktori data ada
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Error: Direktori data tidak ditemukan di {DATA_DIRECTORY}")
        print("Mohon periksa kembali path data Anda.")
    else:
        train_classifier_model(DATA_DIRECTORY, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH)