"""
autocorrect.py
==============
Modul Auto-Correct Cerdas untuk Penerjemah Bahasa Isyarat BISINDO.

Fitur:
1. Exact Match: Jika kata sudah valid, tidak diubah.
2. Character De-duplication: Mengatasi huruf ganda akibat menahan gestur (misal: "BISSINDO" -> "BISINDO").
3. Fuzzy Matching (difflib): Memperbaiki salah eja / huruf selip (misal: "TERMA" -> "TERIMA", "BISNIDO" -> "BISINDO").
4. Priority on Same Length: Jika ada beberapa kandidat, prioritaskan kata dengan panjang yang sama (transposisi huruf).
5. Name & Acronym Guard: Kata pendek (<= 2 huruf) atau nama orang yang tidak cocok tidak dirusak.
"""

import os
import re
import difflib

# Kamus bawaan default (fallback jika file tidak ditemukan)
DEFAULT_VOCAB = {
    'HALO', 'HAI', 'SELAMAT', 'PAGI', 'SIANG', 'SORE', 'MALAM', 'KABAR',
    'TERIMA', 'KASIH', 'SAMA', 'MAAF', 'TOLONG', 'SILAKAN', 'PERMISI', 'SAMPAI', 'JUMPA',
    'BISINDO', 'SIBI', 'ISYARAT', 'BAHASA', 'TULI', 'TUNARUNGU', 'BICARA', 'SUARA',
    'APA', 'SIAPA', 'MANA', 'DIMANA', 'KAPAN', 'MENGAPA', 'KENAPA', 'BAGAIMANA', 'BERAPA',
    'SAYA', 'AKU', 'KAMU', 'DIA', 'KITA', 'KAMI', 'MEREKA', 'ANDA',
    'INI', 'ITU', 'SINI', 'SANA', 'RUMAH', 'SEKOLAH', 'KANTOR', 'TEMPAT', 'JALAN',
    'ADA', 'MAU', 'INGIN', 'SUKA', 'CINTA', 'TAHU', 'PAHAM', 'MENGERTI', 'BISA',
    'MAKAN', 'MINUM', 'TIDUR', 'BANGUN', 'DUDUK', 'BERDIRI', 'JALAN', 'PERGI', 'PULANG',
    'BELAJAR', 'KERJA', 'BANTU', 'BACA', 'TULIS', 'DENGAR', 'LIHAT', 'TANYA', 'JAWAB',
    'BAIK', 'BAGUS', 'SEHAT', 'SAKIT', 'SENANG', 'SEDIH', 'PINTAR', 'BENAR', 'SALAH',
    'MUDAH', 'SUSAH', 'SULIT', 'CEPAT', 'LAMBAT', 'BESAR', 'KECIL', 'BARU', 'LAMA',
    'SEKARANG', 'HARI', 'BESOK', 'KEMARIN', 'JAM', 'WAKTU', 'TAHUN', 'BULAN', 'MINGGU',
    'SATU', 'DUA', 'TIGA', 'EMPAT', 'LIMA', 'ENAM', 'TUJUH', 'DELAPAN', 'SEMBILAN', 'SEPULUH',
    'AYAH', 'IBU', 'BAPAK', 'MAMA', 'PAPA', 'ANAK', 'KAKAK', 'ADIK', 'TEMAN', 'GURU',
    'DAN', 'ATAU', 'TAPI', 'KARENA', 'UNTUK', 'DENGAN', 'DARI', 'KE', 'PADA', 'DI',
    'YA', 'TIDAK', 'BUKAN', 'BELUM', 'SUDAH', 'JANGAN', 'HARUS', 'BOLEH', 'SANGAT'
}


class BisindoAutoCorrect:
    def __init__(self, dictionary_path=None):
        self.words_set = set(DEFAULT_VOCAB)
        self.words_list = sorted(list(self.words_set))

        if dictionary_path and os.path.exists(dictionary_path):
            self.load_dictionary(dictionary_path)

    def load_dictionary(self, dictionary_path):
        """Memuat kamus kata dari file teks."""
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        word = line.upper()
                        self.words_set.add(word)
            self.words_list = sorted(list(self.words_set))
            print(f"[AutoCorrect] Kamus aktif dengan {len(self.words_set)} kata.")
        except Exception as e:
            print(f"[AutoCorrect] Peringatan: Gagal memuat {dictionary_path}: {e}")

    def correct(self, word, similarity_cutoff=0.70):
        """
        Mengoreksi typo kata.
        Mengembalikan tuple: (kata_hasil, is_corrected, similarity_score)
        """
        if not word:
            return "", False, 1.0

        raw_word = word.strip().upper()

        # 1. Exact Match
        if raw_word in self.words_set:
            return raw_word, False, 1.0

        # 2. Kata sangat pendek (<= 2 huruf) jangan diubah sembarangan
        if len(raw_word) <= 2:
            return raw_word, False, 1.0

        # 3. De-duplikasi karakter berulang (misal: "BISSINDO" -> "BISINDO")
        collapsed_word = re.sub(r'(.)\1+', r'\1', raw_word)
        if collapsed_word in self.words_set:
            return collapsed_word, True, 0.95

        # 4. Pencarian Kemiripan (Fuzzy Matching)
        matches = difflib.get_close_matches(
            raw_word,
            self.words_list,
            n=4,
            cutoff=similarity_cutoff
        )

        if matches:
            # Urutkan kandidat: utamakan yang memiliki panjang karakter yang sama (transposisi huruf)
            # lalu berdasarkan skor kemiripan
            best_match = None
            best_score = -1.0

            for m in matches:
                if abs(len(m) - len(raw_word)) > 2:
                    continue
                score = difflib.SequenceMatcher(None, raw_word, m).ratio()
                # Berikan sedikit bonus skor jika panjangnya sama persis
                if len(m) == len(raw_word):
                    score += 0.08
                if score > best_score:
                    best_score = score
                    best_match = m

            if best_match:
                final_ratio = difflib.SequenceMatcher(None, raw_word, best_match).ratio()
                return best_match, True, final_ratio

        # 5. Coba pencarian kemiripan dari bentuk de-duplikasi
        if collapsed_word != raw_word:
            matches_col = difflib.get_close_matches(
                collapsed_word,
                self.words_list,
                n=3,
                cutoff=similarity_cutoff
            )
            if matches_col:
                best_m = matches_col[0]
                if abs(len(best_m) - len(collapsed_word)) <= 2:
                    ratio = difflib.SequenceMatcher(None, collapsed_word, best_m).ratio()
                    return best_m, True, ratio

        # 6. Fallback: Pertahankan kata asli jika tidak ada yang cocok
        return raw_word, False, 0.0


# Uji Coba Singkat
if __name__ == "__main__":
    corrector = BisindoAutoCorrect()
    test_words = [
        "BISINDO",
        "BISNIDO",
        "BIISINDO",
        "TERMA",
        "KASIHH",
        "SAYAA",
        "KMAU",
        "BUDI",
        "YA",
        "SLMAT"
    ]

    print("Hasil Uji Coba Auto-Correct:")
    for w in test_words:
        res, is_corr, score = corrector.correct(w)
        status = f"-> DIKOREKSI: {res} ({score*100:.1f}%)" if is_corr else f"-> ASLI: {res}"
        print(f"  {w:10} {status}")
