# 🔐 HASIL FASE 4: PEMBUATAN DUAL BRANKAS (PRE-TRAIN & FINE-TUNE)

Fase 4 adalah proses krusial di mana data bersih dari Fase 3 dipisahkan secara struktural menjadi dua kantong memori utama (Dual Brankas). Strategi ini dibuat khusus untuk memfasilitasi arsitektur AI tingkat lanjut yang membutuhkan pra-latihan panjang sebelum penyesuaian akhir di dunia nyata.

---

## 🎯 1. Aturan Sakral (Target Deteksi Ekstrem)
Sistem peringatan dini tidak peduli dengan gerimis. Fokus utama model ini adalah mendeteksi ancaman nyata.
- **Batas Ekstrem (Treshold):** `> 100 mm / hari` (Kategori: Hujan Sangat Lebat / Badai)
- Data yang melampaui batas ini diberi label bahaya **1 (EXTREME)**, sisanya diberi label **0**.

---

## 🚀 2. Strategi Arsitektur "RAM-Safe" & Konversi Zona Waktu
Untuk mencegah *Crash* akibat kehabisan memori (*Out of RAM*) saat mengolah file netCDF master:
1. Data satelit ERA5 diekstrak per stasiun *sebelum* dimuat secara penuh ke memori Pandas. Proses penarikan menggunakan **Bilinear Interpolation** untuk mendapatkan letak geografis yang sangat akurat di 5 Stasiun BMKG.
2. Waktu universal satelit (GMT) digeser ke waktu lokal Jakarta **WIB (GMT+7)**. Ini adalah SOP resmi BMKG agar pembacaan hujan sesuai dengan waktu lokal terjadinya bencana.

---

## 📦 3. Konstruksi Dual Brankas & Hasil Matriks

Data dibelah menjadi dua segmen waktu yang sangat tegas.

### 🛡️ Brankas 1: Pre-Training (Satelit Murni)
Digunakan untuk membangun insting dasar model AI terhadap dinamika cuaca Jabodetabek selama 8 tahun tanpa jeda.
- **Rentang Waktu:** 2016 hingga Mei 2024
- **Komposisi:** Murni data iklim satelit (ERA5-Land)
- **Output:** `brankas1_pretrain.parquet`
- **Dimensi Akhir:** **15.370 baris** (21 kolom)

### 💎 Brankas 2: Fine-Tuning (Fusi BMKG Ground-Truth)
Digunakan sebagai "Buku Kunci Jawaban Resmi". Di sini, data satelit bersatu dengan data observasi hujan riil stasiun BMKG.
- **Rentang Waktu:** Juni 2024 hingga Mei 2026
- **Komposisi:** Data Satelit (ERA5) + Data Lapangan (BMKG)
- **Output:** `brankas2_finetune.parquet`
- **Dimensi Akhir:** **3.625 baris** (21 kolom)

---

## 🛡️ 4. Inovasi "Titanium Shield" & Laporan Ketimpangan

Sebuah mekanisme pertahanan (*Titanium Shield*) diaktifkan pada akhir program untuk memastikan bahwa **Brankas 1 dan Brankas 2 memiliki susunan kolom yang identik 100% (21 Kolom)**. Kolom-kolom BMKG yang tidak ada di tahun-tahun awal satelit disuntikkan dengan nilai kosong secara simetris, mencegah malfungsi arsitektur neural network di Fase 6.

### 📊 Laporan Ketimpangan Data (Imbalance Ratio)
Dari visualisasi histogram `RR (mm)`, terlihat jelas sifat alamiah hujan ekstrem:
- Pada **Brankas 1**, rasio kejadian hari aman melawan hari badai (>100mm) adalah **1 : 3**. (Kondisi cukup merata).
- Pada **Brankas 2** (Dunia Nyata BMKG), dari ribuan hari, badai ekstrem hanya muncul **21 kali** (Rasio ketimpangan luar biasa ekstrem **1 : 172**).

Fenomena kelangkaan ekstrem inilah yang akan diselesaikan secara algoritmik di Fase 6 (Model Pelatihan) menggunakan *Loss Function* berbobot denda.
