# 📖 BUKU LOG EKSPERIMEN: PRAPROSES & REKAYASA DATA (DATA ENGINEERING)

Dokumen ini adalah jurnal ilmiah yang mencatat seluruh jejak rekam uji coba (*trial and error*) dalam proses pembersihan, fusi, dan augmentasi data. Pemisahan log ini bertujuan untuk membuktikan secara empiris tingginya kompleksitas dan kedalaman analisis data (*Data Engineering*) sebelum masuk ke tahap pemodelan AI.

---

## 🧪 DAFTAR EKSPERIMEN PRAPROSES DATA

| No. Eksperimen | Tanggal | Deskripsi / Hipotesis Praproses | Parameter yang Diubah | Hasil & Metrik Evaluasi | Kesimpulan / Tindakan Selanjutnya |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PP-001** | *Juni 2026* | **Penambalan Nilai Kosong BMKG** | Interpolasi Linear, Ffill, Bfill pada nilai anomali 8888 & 9999. | Data BMKG bersih tanpa NaN. Penggabungan *Inner Join* menghasilkan 68.976 baris Hibrida yang solid. | Terapkan metode interpolasi `ffill().bfill()` secara permanen di Brankas 2. |
| **PP-002** | *Juni 2026* | **Pencarian Rentang Waktu Historis (Time-Split)** | Brankas 1 (2016 - Mei 2024), Brankas 2 (Juni 2024 - Mei 2026). | Total Brankas 1: 368.880 baris. Total Brankas 2: 68.976 baris. Tidak ada *Data Leakage*. | Pertahankan arsitektur *Dual Brankas*. |
| **PP-003** | *Juni 2026* | **Eksperimen Panjang Window (Tensor 72-Jam vs 336-Jam)** | `SEQ_LENGTH` = 72 (3 Hari) vs `SEQ_LENGTH` = 336 (14 Hari). | Tensor 72-jam cepat diproses, namun kurang merekam musim. Tensor 336-jam membutuhkan teknik Anti RAM-Crash (Sliding 24-jam). | **Pilih 336-Jam (14 hari)** karena cuaca ekstrem Jabodetabek butuh histori panjang. |
| **PP-004** | *Juni 2026* | **Perhitungan Threshold Klasifikasi Bencana** | Menghitung hujan maksimum historis (250.3 mm) dan melakukan normalisasi MinMaxScaler. | Ditemukan batas Siaga Hujan Lebat (>20mm) = `0.0799`. Batas Ekstrem (>50mm) = `0.1998`. | Threshold normalisasi ini digunakan sebagai pijakan Klasifikasi *BCE Loss* di AI. |
| **PP-005** | *Juni 2026* | **Evaluasi RAM Tensor Generator** | Memuat 50.000 sampel awal dengan *Partial Fit* untuk `MinMaxScaler` sebelum proses stasiun. | Berhasil menyimpan memori (*RAM Colab tidak jebol*). | Metode *Streaming per Stasiun* dengan *Partial Fit* jadi SOP standar. |

---

## 📌 CATATAN PENTING & DISKUSI PENEMUAN
* **Anomali Sensor BMKG:** Terdapat nilai `8888` dan `9999` yang merusak skala normalisasi data, akhirnya ditangkap sebagai `NaN` lalu di-interpolasi secara linear.
* **Krisis RAM Colab:** Saat mengonversi 368.880 baris menjadi `[15305, 336, 11]` tensor 3D/4D, RAM langsung meledak. Trik pembacaan per-stasiun *(streaming)* dan konversi ke `np.float32` terbukti ampuh.
* **Hujan Ekstrem Tertinggi:** Dari pencarian ekstremitas, tercatat hujan setinggi 250.3 mm/hari pernah melanda Jabodetabek. Data ekstrem ini digunakan untuk menghitung nilai skala konversi biner AI.
