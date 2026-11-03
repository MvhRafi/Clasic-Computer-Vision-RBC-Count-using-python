# Pemisahan dan Penghitungan Sel Darah Merah Bersentuhan Menggunakan Transformasi Jarak dan Algoritma Watershed

Dokumen ini merupakan petunjuk penggunaan dan panduan reproduksi untuk **Topik 3: Penghitungan Sel Darah Merah Bersentuhan** pada Mata Kuliah Teori & Praktikum Computer Vision (D4 Teknik Elektronika).

Proyek ini bertujuan untuk mendeteksi, memisahkan, dan menghitung jumlah sel darah merah (*Red Blood Cells* / RBC) pada citra mikroskopis yang saling menempel (*cluttered/overlapping*) menggunakan pendekatan **Computer Vision Tradisional** tanpa model *Deep Learning*.

## 📌 Fitur Utama Pipeline

1. **Masking & Isolasi Leukosit (WBC)**: Pemisahan sel darah putih beserta sitoplasmanya menggunakan kombinasi ruang warna HSV dan LAB agar tidak mengganggu estimasi latar belakang.

2. **Koreksi Pencahayaan (Background Subtraction)**: Perataan intensitas pencahayaan mikroskop menggunakan subtraksi morfologi pada kanal Hijau (*Green Channel*).

3. **Pengisian Lubang Sel (Hole Filling)**: Rekonstruksi biner untuk menggeneralisasi cekungan bikonkaf di tengah sel darah merah.

4. **Pemisahan Objek Bersentuhan (Watershed & Distance Transform)**: Penggunaan peta jarak Euclidean L2 dan puncak lokal (*local maxima seeds*) untuk memisahkan kontur sel yang saling menempel.

5. **Evaluasi Otomatis Berbasis CSV**: Pengujian dinamis menggunakan dataset terisolasi (*Tuning vs Test*) dengan metrik evaluasi *Mean Absolute Error* (MAE) dan Bias.

## 🛠️ Kebutuhan Perangkat Lunak & Library

Proyek ini dikembangkan menggunakan bahasa pemrograman **Python 3.8+** dan beberapa pustaka pengolahan citra serta analisis data:

* **Python 3.8 / 3.9 / 3.10 / 3.11**

* **OpenCV (`opencv-python`)**: Pengolahan citra dasar, morfologi, *Connected Components*, dan *Distance Transform*.

* **NumPy (`numpy`)**: Manipulasi matriks dan array multidimensi.

* **Pandas (`pandas`)**: Pembacaan file CSV ground truth dan penyusunan tabel evaluasi.

* **scikit-image (`skimage`)**: Ekstraksi benih *local maxima* (`peak_local_max`) dan segmentasi *Watershed*.

### Perintah Instalasi Dependensi

Buka terminal / Command Prompt Anda, lalu jalankan perintah berikut:

```
pip install opencv-python numpy pandas scikit-image

```

## 📁 Struktur File & Direktori Proyek

```
├── Data Latih Blood Cell.csv    # File CSV Ground Truth (filename, gt_count, split)
├── test_rbc_2.py                 # Skrip utama pipeline inferensi & pengujian dataset uji
├── tuning.py                     # Skrip Grid Search tuning hyperparameter (opsional)
├── config_terkunci.json          # File Simpanan parameter terbaik hasil tuning
├── README.md                     # Dokumentasi petunjuk penggunaan
└── output_visual/                # Folder otomatis tempat menyimpan citra penanda & masker
    ├── penanda_BloodImage_*.jpg  # Citra visualisasi kontur hijau & titik benih merah
    ├── mask_utuh_BloodImage_*.jpg # Citra masker biner hasil hole-filling
    └── seed_BloodImage_*.jpg      # Citra titik puncak benih lokal maxima

```

## 📊 Format File Data CSV (`Data Latih Blood Cell.csv`)

File CSV harus berada dalam direktori proyek dan memiliki format header kolom sebagai berikut:

```
Nama,Acuan Manual,Split
BloodImage_00339,24,tuning
BloodImage_00340,22,tuning
BloodImage_00343,19,tuning
BloodImage_00351,26,test
BloodImage_00355,24,test
BloodImage_00357,23,test
BloodImage_00359,22,test
BloodImage_00366,21,test

```

> **Catatan**: Kolom `Split` harus bernilai `tuning` (untuk data latih/penyetelan) atau `test` (untuk data uji terisolasi).

## 🚀 Langkah-Langkah Menjalankan Program

### 1. Pengisian Path Direktori Citra

Buka file `test_rbc_2.py` (dan `tuning.py` jika digunakan), pastikan variabel `BASE` atau `TEST_IMG_FOLDER` sudah mengarah ke folder direktori tempat citra `.jpg` Anda disimpan:

```
BASE = path file
TEST_IMG_FOLDER = os.path.join(BASE, "Testing", "Images")
CSV_PATH = "Data Latih Blood Cell.csv"

```

### 2. (Opsional) Penyetelan Parameter (`tuning.py`)

Jika Anda ingin melakukan eksperimen *Grid Search* untuk mencari kombinasi parameter baru pada subset data `tuning`:

```
python tuning.py

```

*Skrip ini akan mengeksekusi iterasi parameter, mencetak metrik MAE/Bias terbaik, dan mengunci konfigurasi ke `config_terkunci.json`.*

### 3. Eksekusi Pengujian Utama (`test_rbc_2.py`)

Untuk menjalankan pemrosesan citra pada dataset uji terisolasi (`test`) dan menghasilkan tabel evaluasi serta visualisasi:

```
python test_rbc_2.py

```

## 📈 Metrik Evaluasi Akhir

Kinerja pemisahan sel diukur menggunakan dua metrik statistik terhadap acuan manual (*Ground Truth*):

1. **Mean Absolute Error (MAE)**:
   

   $$
   \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \vert{}y_i - \hat{y}_i\vert{}
   $$

   
   *Mengukur rata-rata selisih mutlak jumlah sel hasil prediksi (*$\hat{y}_i$*) terhadap acuan manual (*$y_i$*).*

2. **Bias Perhitungan**:
   

   $$
   \text{Bias} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
   $$

   
   *Menunjukkan kecenderungan algoritma apakah mengalami Overcounting (*$\text{Bias} > 0$*) atau Undercounting (*$\text{Bias} < 0$*).*

## 📤 Output Program

Setelah `test_rbc_2.py` selesai dijalankan:

1. **Terminal**: Menampilkan tabel perbandingan hasil per citra uji, nilai galat *Baseline*, nilai galat *Watershed*, serta ringkasan MAE dan Bias akhir.

2. **File `hasil_pengujian_utama.csv`**: Tersimpan otomatis berisi tabel lengkap ringkasan pengujian.

3. **Folder `output_visual/`**: Berisi citra hasil menggambar kontur sel (garis hijau), titik benih pusat sel (titik merah), dan angka nomor indeks sel.

## ⚖️ Lisensi & Pernyataan Batas Lingkup

Program ini dikembangkan terbatas untuk keperluan pemenuhan Tugas Berbasis Masalah Mata Kuliah Computer Vision. **Tidak ditujukan untuk penggunaan diagnosis medis langsung atau sistem inspeksi klinis otomatis tanpa supervisi profesional.**
