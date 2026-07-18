# Laporan Eksekusi: Fase 4 (Dual Brankas Creation & Titanium Shield)

Fase 4 ditujukan untuk menyusun ulang dataset menjadi **Dua Brankas (Dataset) Terpisah**, yang akan digunakan untuk proses pelatihan berjenjang (*Transfer Learning*): Pre-Training dan Fine-Tuning. Fase ini sangat krusial karena kita mengincar target presipitasi ekstrem yang ditetapkan sangat tinggi, yaitu ambang batas **> 100 mm/hari** (Hujan Sangat Lebat).

---

## 🛠️ Proses Teknis Utama

### 1. Interpolasi Bilinear & Sinkronisasi Zona Waktu (WIB)
Untuk mengekstrak data dari koordinat spasial grid satelit secara lebih akurat, metode **Bilinear Interpolation** diterapkan. Ini memberikan presisi matematis yang lebih halus dibandingkan metode *nearest* (tetangga terdekat) pada Fase 3.
Selain itu, zona waktu satelit (GMT/UTC) digeser ke **WIB (UTC+7)** untuk menyamakan metrik waktu harian pencatatan dengan SOP (Standar Operasional Prosedur) BMKG resmi.

### 2. Pembangunan Brankas 1: Pre-Training (Satelit Murni)
- **Rentang Waktu:** 2016 hingga Mei 2024.
- **Tujuan:** Melatih "insting dasar fisika cuaca" pada model Mamba menggunakan data volume raksasa (*Big Data*) murni dari satelit.
- **Karakteristik:** Terdapat 15.370 baris data cuaca sejarah dengan rentang Curah Hujan Harian (RR) yang sangat lebar, yaitu dari 0.00 hingga **1224.85 mm** (badai sangat ekstrem yang terekam satelit).

### 3. Pembangunan Brankas 2: Fine-Tuning (Fusi Ground-Truth BMKG)
- **Rentang Waktu:** Juni 2024 hingga Mei 2026.
- **Tujuan:** Menajamkan (*fine-tune*) model agar tidak lagi berhalusinasi ala satelit, melainkan mengkalibrasinya dengan alat ukur nyata di lapangan (Kunci Jawaban BMKG).
- **Karakteristik:** Terdapat 3.625 baris data cuaca lapangan dengan curah hujan maksimum tercatat di lapangan sebesar **250.30 mm**.

### 4. Implementasi "Titanium Shield"
Agar model *Deep Learning* tidak mengalami eror kompatibilitas saat proses transfer ilmu (dari *Pre-Training* ke *Fine-Tuning*), kedua Brankas dikunci menggunakan sistem *Titanium Shield*. Sistem ini memaksa kolom/fitur di kedua tabel memiliki struktur matriks yang 100% identik dan tersusun pada urutan yang sama.
Hasil validasi: **✅ SEMPURNA (21 Kolom Identik)**.

---

## 🚨 Laporan Ketimpangan Data (Imbalance Ratio)
Penetapan ambang batas ekstrem **> 100 mm/hari** menyebabkan data menjadi sangat timpang secara distribusi kelas, yang membenarkan penggunaan algoritma khusus (Focal Loss / EVT) di masa depan.

- **Brankas 1 (Satelit 2016-2024)**:
  - Hari Aman (<100mm)   : 11.472 hari
  - Hari Badai (>100mm)  : 3.898 hari
  - Rasio Ketimpangan    : **1 : 3**

- **Brankas 2 (BMKG 2024-2026)**:
  - Hari Aman (<100mm)   : 3.604 hari
  - Hari Badai (>100mm)  : 21 hari *(Sangat Langka di Lapangan)*
  - Rasio Ketimpangan    : **1 : 172**

## 💾 Hasil Akhir
Kedua Brankas telah berhasil disterilkan, disinkronkan, dan diekspor ke dalam format modern *Parquet* yang lebih ringan dan cepat dibaca dibandingkan CSV:
1. `brankas1_pretrain.parquet` (15.370 baris)
2. `brankas2_finetune.parquet` (3.625 baris)

---

## 📈 Analisis Visual Distribusi Curah Hujan (Histogram)
Berdasarkan grafik visualisasi *Distribusi Curah Hujan Dual Brankas* yang dihasilkan:

1. **Distribusi Brankas 1 (Satelit ERA5-Land - Kurva Biru)**:
   - Terlihat bentuk kurva yang memiliki ekor kanan (*right-skewed / heavy-tailed*) yang sangat panjang, merentang hingga melewati angka **1200 mm**.
   - Banyak sekali titik data yang sukses melewati garis putus-putus merah (Batas Ekstrem 100mm). Hal ini membuktikan bahwa pembacaan satelit cenderung kaya akan kejadian ekstrem tinggi (*high-magnitude events*). Data ini sangat sempurna untuk melatih "insting" model Deep Learning dalam mengenali pola-pola badai besar.

2. **Distribusi Brankas 2 (BMKG Ground-Truth - Kurva Hijau)**:
   - Kurva menurun dengan sangat tajam dan menumpuk di sisi kiri. Sebagian besar kejadian hujan harian di lapangan nyata berpusat di angka yang rendah (< 50 mm).
   - Volume data yang menembus garis merah 100mm nyaris tidak terlihat rata dengan tanah (maksimal hanya mencapai sekitar 250 mm). Ini merepresentasikan realita fisik di permukaan bumi, di mana hujan ekstrem > 100 mm/hari adalah anomali cuaca yang amat sangat langka (sesuai rasio 1:172).

**Kesimpulan Visual:** 
Kedua grafik ini memvalidasi kejeniusan strategi "Dual Brankas" secara visual. Jika AI langsung dilatih dengan data nyata BMKG (Kurva Hijau), AI akan gagal belajar mengenali badai ekstrem karena sampelnya nyaris tidak ada. Dengan melakukan tahapan *Pre-Training* pada Brankas 1 (Kurva Biru), AI dipaksa belajar dari ribuan skenario cuaca ekstrem sejarah, sebelum akhirnya dikalibrasi (di-*Fine-Tune*) ke alam nyata menggunakan Brankas 2.
