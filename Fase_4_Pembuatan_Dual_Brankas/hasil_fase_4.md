# 🔒 HASIL FASE 4: DUAL BRANKAS & TITANIUM SHIELD

Fase 4 berfungsi sebagai sistem arsitektur pengamanan data dan sinkronisasi lanjutan. Tabel raksasa hasil hibridasi dari **Fase 3** diolah di sini untuk memisahkan rentang waktu agar tidak terjadi kebocoran kebocoran informasi masa depan ke masa lalu saat model dilatih (*Data Leakage*), serta menyelaraskan jam fisika antardimensi.

---

## 📥 INPUT DARI FASE 3
- File tunggal yang bersih: `dataset_satelit_stasiun_bersih.parquet` yang berisi integrasi data satelit (Era5-Land) dan Stasiun BMKG.

---

## 🛡️ LOGIKA KODE & ALGORITMA PENGAMANAN

Pada Fase 4 ini, sistem menetapkan standar operasional cuaca yang sangat ketat:

### 1. Sinkronisasi Zona Waktu (GMT ke WIB)
Satelit Eropa merekam waktu berdasarkan GMT (Universal Time), sementara BMKG mencatat berbasis Waktu Indonesia Barat (WIB). Jika tidak dikoreksi, pola mendung satelit akan tertinggal 7 jam dari badai aslinya di daratan Jakarta. 
Skrip memperbaiki ini dengan menggeser (*Shift*) tanggal dan jam satelit menjadi **GMT+7** sebelum dilakukan *resampling* harian.

### 2. Algoritma RAM-Safe (Bilinear Interpolation per Stasiun)
Untuk mencegah RAM Google Colab meledak karena mengkalkulasi jutaan titik secara bersamaan, skrip menggunakan metode *Bilinear Interpolation* yang mengekstrak nilai piksel cuaca satelit langsung secara terfokus pada titik koordinat persis setiap stasiun, stasiun demi stasiun.

### 3. Penetapan "Threshold Sakral" (Badai Ekstrem >= 50 mm)
Data dipindai untuk mencari target anomali tertinggi: Curah Hujan Ekstrem (di atas 50 mm/hari). Laporan Fase 4 menemukan ketimpangan kelas badai yang cukup brutal. Artinya, badai mematikan ini sangat langka (minoritas super), yang mengharuskan algoritma *SMOTETomek* untuk turun tangan nanti di Fase 5.

### 4. Pembentukan "Titanium Shield" (Split Data Aman)
Tabel raksasa dipecah menjadi dua rentang waktu absolut (Dual Brankas) berstruktur pelindung dengan presisi (11 fitur cuaca satelit ERA5-Land + 6 indikator stasiun BMKG = Total 17 fitur persilangan).
- **Brankas 1 (Pre-Train):** Rentang **Tahun 2016 hingga Mei 2024** (Bahan ajar utama satelit murni sebelum sensor BMKG tersedia).
- **Brankas 2 (Fine-Tune / Test):** Rentang **Juni 2024 hingga Mei 2026** (Data fusi hibrida satelit dan sensor BMKG aktual untuk kalibrasi mutlak).

---

## ➡️ OUTPUT UNTUK FASE SELANJUTNYA
Bahan ajar telah diamankan ke dalam dua *Vault* yang terpisah sempurna:
1. `brankas1_pretrain.parquet`
2. `brankas2_finetune.parquet`

Kedua *brankas* 2 Dimensi ini **dikirim menuju Fase 5 (4D Spatio-Temporal Windowing)**. Di sana, data datar ini akan dilipat secara matematis ke dalam ruang 4 Dimensi, dan kelemahannya (kelangkaan data badai) akan disembuhkan oleh algoritma SMOTE.
