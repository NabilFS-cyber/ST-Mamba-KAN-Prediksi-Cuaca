# 📐 HASIL FASE 5: DATA WINDOWING 4D + SMOTE (VERSI BERSIH & BENAR)

Dokumen ini merangkum proses pengubahan struktur data tabular 2D menjadi tensor 4D murni yang sangat krusial bagi arsitektur GNN-Mamba, sekaligus menjelaskan penyelesaian cacat algoritma (bug) *Data Leaking* yang sempat terjadi saat SMOTE.

---

## 📥 1. Input Data
- **File Input:** 6 File pickle dari Fase 4 (`b1_train.pkl`, `b2_train.pkl`, dst).
- **Format Awal:** Tabel data 2D per stasiun.

---

## 🔧 2. Proses Windowing 4D (Sliding Window)

Arsitektur canggih (seperti GNN dan Mamba) tidak bisa membaca data sebaris demi sebaris. Mereka memerlukan informasi "sejarah cuaca masa lalu" sekaligus "peta letak stasiun".
Oleh karena itu, data diubah strukturnya menjadi Tensor PyTorch 4 Dimensi dengan konfigurasi:
`[N_samples, 14, 5, 18]`

Penjelasan Dimensi:
- `N_samples` = Jumlah baris/kejadian hari hujan.
- `14` = **Time Steps** (Model akan selalu melihat rekam jejak cuaca 14 hari ke belakang).
- `5` = **Stations** (Kelima stasiun direkatkan menjadi satu matriks spasial yang akan dibaca oleh GAT/GNN).
- `18` = **Features** (Ke-18 variabel iklim satelit).

---

## 🚨 3. Perbaikan Kritis: SMOTE Data Leaking Bug

Pada arsitektur eksperimen sebelumnya, terjadi *Bug* parah: RMSE regresi melonjak di atas 100 mm. 
**Penyebab:** Fungsi SMOTE (teknik *oversampling* untuk menyeimbangkan kelas minoritas) yang tadinya hanya diterapkan pada kelas klasifikasi (`yc`) ternyata mengacak-acak urutan label regresi (`yr`). Akibatnya, tensor cuaca dipasangkan dengan milimeter hujan yang salah.

**Solusi Resolusi (Fase 5B):** 
Sebelum dimasukkan ke algoritma SMOTE, label regresi (`yr`) **diikat / di-stack** secara absolut bersamaan dengan matriks fitur 4D (yang dipipihkan sementara). Dengan pengikatan (`column_stack`) ini, ketika SMOTE menggandakan data kelas Siaga/Waspada, label milimeter hujannya ikut tergandakan dengan korelasif fisik yang 100% terjaga.

---

## ⚖️ 4. Distribusi Kelas (Hasil SMOTE)

Karena kelas badai (Siaga dan Waspada) sangat langka di dunia nyata, SMOTE sangat diwajibkan **hanya pada Training Set**.
Perbandingan sebelum dan sesudah:

- **Sebelum SMOTE:**
  - Aman: ~70% (Mayoritas absolut)
  - Siaga: ~20%
  - Waspada: ~10% (Paling sulit dikenali)

- **Sesudah SMOTE:**
  - Aman: ~33%
  - Siaga: ~33%
  - Waspada: ~33%
  *(Ketiga kelas kini seimbang, mencegah model AI menjadi malas/bias menebak "Aman" terus-menerus).*

---

## 📦 5. Output Tensor Files

Data 4D yang telah steril dan seimbang diekspor dalam format tensor asli PyTorch (`.pt`). Ini membuat pemuatan data saat *Training* di Fase 6 menjadi seketika.

Lokasi Penyimpanan: `/content/drive/MyDrive/Riset_ERA5_Land/tensors_mamba/`
(Terdapat 18 file, 3 bagian per brankas per split):
- Fitur Iklim: `_X_4d.pt` (Tensor `[N, 14, 5, 18]`)
- Label Regresi: `_yr_4d.pt` (Diubah secara logaritmik `log1p` agar nilai ekstrem tidak merusak gradien)
- Label Klasifikasi: `_yc_4d.pt`

Seluruh tensor siap dikonsumsi mentah-mentah oleh model **ST-Mamba-KAN**!
