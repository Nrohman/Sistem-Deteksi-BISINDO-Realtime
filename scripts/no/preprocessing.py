import tensorflow as tf
import os

def load_and_preprocess_image_dataset(data_dir, img_height, img_width, batch_size, validation_split=0.2, subset='training', seed=123):
    """
    Memuat dataset gambar dari direktori, mengubah ukurannya, dan menormalisasi piksel.
    Mengembalikan dataset yang sudah dinormalisasi dan nama-nama kelas.
    """
    # Langkah 1: Muat dataset asli untuk mendapatkan class_names
    # Jangan langsung normalisasi di sini
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        validation_split=validation_split,
        subset=subset,
        seed=seed # Untuk reproduksibilitas
    )

    class_names = dataset.class_names # Simpan nama kelas di sini

    # Langkah 2: Normalisasi piksel dari [0, 255] ke [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    return normalized_dataset, class_names # Kembalikan juga class_names

if __name__ == "__main__":
    # Path ke folder dataset Anda setelah diekstrak
    # Sesuaikan path ini jika berbeda!
    data_directory = "data"

    # Periksa apakah folder ada
    if not os.path.exists(data_directory):
        print(f"Error: Direktori data tidak ditemukan di {data_directory}")
        print("Pastikan Anda sudah mengunduh dan mengekstrak dataset Kaggle.")
        print("Struktur yang diharapkan: project_bisindo/data/Indonesian Sign Language (BISINDO) Recognition/A, B, C, dst.")
    else:
        IMG_HEIGHT = 224 # Ukuran standar untuk MobileNetV2
        IMG_WIDTH = 224
        BATCH_SIZE = 32

        print("Memuat dataset pelatihan...")
        train_ds, train_class_names = load_and_preprocess_image_dataset(
            data_directory, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
            validation_split=0.2, subset='training'
        )

        print("Memuat dataset validasi...")
        val_ds, val_class_names = load_and_preprocess_image_dataset(
            data_directory, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE,
            validation_split=0.2, subset='validation'
        )

        # Pastikan nama kelas sama untuk pelatihan dan validasi (seharusnya selalu sama)
        class_names = train_class_names

        print(f"Jumlah kelas: {len(class_names)}")
        print(f"Nama kelas: {class_names}")

        # Contoh mengambil satu batch
        for images, labels in train_ds.take(1):
            print(f"Bentuk gambar batch: {images.shape}")
            print(f"Bentuk label batch: {labels.shape}")
            break

        print("\nDataset berhasil dimuat dan dipra-proses.")
        print("Langkah selanjutnya adalah melatih model.")

        # Anda bisa menyimpan nama kelas (huruf) ke file untuk digunakan nanti
        # Pastikan folder 'dictionary' ada
        dictionary_folder = "../dictionary"
        os.makedirs(dictionary_folder, exist_ok=True) # Buat folder jika belum ada

        class_names_path = os.path.join(dictionary_folder, "class_names.txt")
        with open(class_names_path, "w") as f:
            for name in class_names:
                f.write(name + "\n")
        print(f"Nama kelas disimpan di {class_names_path}")