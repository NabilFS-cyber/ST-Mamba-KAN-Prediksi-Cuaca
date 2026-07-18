# ⛈️ HASIL FASE 7B: THE ELITE MASTERPIECE (TRUE 4D GNN + ST-MAMBA)

Dokumen ini merangkum penjelasan arsitektur *code* dan hasil pelatihan akhir dari **Fase 7B**, yang merupakan iterasi tertinggi (The Elite Masterpiece) dari proyek AI Prediksi Cuaca Ekstrem.

---

## 📌 1. BEDAH ARSITEKTUR (CODE EXPLANATION)

### A. Format Tensor True 4D (`[Batch, Time, Station, Features]`)
Tidak seperti Fase awal yang memipihkan (*flatten*) data spasial, Fase 7B mempertahankan bentuk asli alam semesta dalam format 4 dimensi. 
- `Batch` = Jumlah sampel.
- `Time` = 14 hari ke belakang.
- `Station` = 5 stasiun BMKG (Tanjung Priok, Kemayoran, Halim, Citeko, Pondok Betung).
- `Features` = 18 variabel cuaca (Suhu, Angin, Curah Hujan, dll).

### B. Spatial Graph Neural Network (True GNN)
Model menggunakan matriks jarak geospasial aktual (Haversine) beradius 20 km. Perkalian *graph* dilakukan secara presisi matematis tanpa merusak dimensi waktu menggunakan operasi tensor tingkat lanjut:
```python
# Perkalian Graph 4 Dimensi
x = torch.einsum('btsf, sg -> btgf', x, self.adj_matrix)
```

### C. Mamba State-Space Model
Setelah fitur spasial (GNN) terkumpul, data dilempar ke blok Mamba. Mamba menggantikan LSTM untuk "mengingat" pola temporal selama 14 hari dengan kecepatan linear (jauh lebih cepat dan memorinya tidak bocor untuk urutan panjang).

---

## 🧠 2. THE ELITE LOSSES (FUNGSI HUKUMAN)

Kode Fase 7B menggunakan kombinasi 3 *Loss Function* mutakhir tingkat Jurnal Q1:

1. **PINN (Physics-Informed Neural Network) Loss:**
   Memaksa model untuk patuh pada **Hukum Konservasi Massa Kelembapan Udara**. Jika prediksi cuaca melanggar hukum fisika alam (misal uap air tidak sebanding dengan curah hujan), model akan didenda.

2. **EVT (Extreme Value Theory) Loss:**
   Hukuman eksponensial khusus untuk kelas Regresi. Jika model *under-predict* (meremehkan) badai besar yang curah hujannya jauh di atas rata-rata (ekstrem), nilai *loss* akan dilipatgandakan secara drastis. Ini memaksa RMSE turun tajam pada kejadian badai.

3. **Ordinal Cost Focal Loss:**
   Digunakan pada Klasifikasi. Kesalahan memprediksi "Siaga (Bahaya)" menjadi "Aman" memiliki bobot denda/dosa 3x lipat lebih besar dibanding kesalahan sebaliknya. Ini membangun insting keselamatan pada model.

---

## 🚀 3. HASIL PELATIHAN FINAL (EVALUASI AKHIR)

Model dilatih dengan mekanisme perlindungan ganda: **Snapshot Ensemble** (mengingat 3 iterasi terbaik) dan **TTA (Test-Time Augmentation)**.

### A. Performa Regresi (Ground-Truth BMKG)
* **RMSE Regresi:** `16.53 mm` 
* **Conformal Prediction Bound:** `± 24.55 mm` (90% Confidence Interval)
* *Interpretasi:* Model sangat presisi mengestimasi milimeter curah hujan. Keberadaan batas toleransi (*Conformal*) membuatnya sangat layak dipakai oleh BMKG secara operasional, karena menyertakan perhitungan ketidakpastian.

### B. Performa Klasifikasi (Deteksi Siaga Badai)
* **Akurasi Total:** `88.16%`
* **Balanced Accuracy:** `79.12%`
* **Macro F1-Score:** `0.790`
* **Recall Kelas Siaga (≥50mm):** `0.92` (92%)
* *Interpretasi:* Dari 100 badai ekstrem yang melanda Jabodetabek, model ini sanggup membunyikan alarm pada **92 badai secara akurat**. Ini adalah tingkat keandalan yang luar biasa tinggi untuk Sistem Peringatan Dini Bencana Hidrometeorologi.
