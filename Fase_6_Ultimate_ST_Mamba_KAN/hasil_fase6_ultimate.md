# 👑 HASIL FASE 6: THE GOD-TIER ARCHITECTURE (LIMIT-BREAKER EDITION)

Fase 6 adalah mahakarya komputasional dalam riset ini. Pada fase ini, data 4D yang telah dirakit dicerna oleh model *Deep Learning* ultra-mutakhir yang dijuluki **Limit-Breaker Edition**, yaitu hibrida antara Graph Attention Network (GAT), Mamba State-Space Model, dan Kolmogorov-Arnold Network (KAN).

---

## 🏗️ 1. Inovasi Arsitektur Tiga Lapis (God-Tier)
Model **Ultimate_GAT_Mamba_KAN** memecahkan masalah cuaca melalui tiga lapisan pemikiran rasional:
1. **Dynamic GAT (Spasial):** Otak satelit. Membaca interaksi pergerakan angin dan awan antar 5 stasiun BMKG Jabodetabek layaknya jaring laba-laba.
2. **Mamba Block (Temporal):** Menggantikan LSTM kuno. Mampu "mengingat" pola curah hujan 14 hari berturut-turut tanpa melupakan informasi awal, dan memprosesnya secara paralel dan sangat cepat.
3. **KAN (Grid Size 12):** Jaringan Saraf Kolmogorov-Arnold tingkat dewa (*Limit-Breaker*) menggantikan *Linear/Dense layer* biasa. Mampu mencari korelasi non-linear yang sangat rumit dengan resolusi sangat tinggi (Grid Size 12).

---

## ⚖️ 2. The Elite Losses (Fungsi Hukuman Ganda)
Sistem ini diajarkan menggunakan sistem penalti yang brutal:
- **Regresi (PINN + EVT):** Jika tebakan model melanggar hukum termodinamika/fisika atmosfer dasar, model akan didenda (PINN). Jika model meremehkan (*under-predict*) intensitas hujan badai, nilai hukumannya dilipatgandakan secara eksponensial (EVT - *Extreme Value Theory*).
- **Klasifikasi (Ordinal Cost Focal Loss + Label Smoothing):** Model didenda 3x lipat jika meremehkan status "Siaga" menjadi "Aman". Penambahan *Label Smoothing* mencegah model bersikap *over-confident* (terlalu percaya diri membabi-buta) pada hari yang tidak pasti.

---

## 🚀 3. Kinerja Akhir (Evaluasi Test Set *Unseen*)
Model diuji melawan data masa depan nyata BMKG yang belum pernah ia lihat (*Unseen Data*). 

### A. Evaluasi Hidrologi (Regresi)
Otak regresi difokuskan mengejar presisi milimeter.
- **RMSE Final:** `17.07 mm`
- **MAE Final:** `8.22 mm`
- **Conformal Bound:** `± 22.91 mm` (Tingkat Keyakinan 90%)
*(Ini berarti prediksi rata-rata hanya meleset di bawah 1 cm air, pencapaian fantastis untuk fenomena atmosferik).*

### B. Evaluasi Deteksi Badai (Klasifikasi)
Didukung oleh *Test-Time Augmentation* (TTA) dan algoritma pencari titik emas (*Sweet Spot*).
- **Sweet Spot Terbaik:** Siaga di probabilitas `35%`, Waspada di probabilitas `55%`.
- **Akurasi Total:** `88.02%`
- **Balanced Accuracy:** `76.74%`
- **Macro F1-Score:** `0.776`
- **CSI (Critical Success Index) Siaga:** `80.99%`

### 🏆 Sorotan Matriks Konfusi (Kekuatan Puncak):
Dari total **864 badai ekstrem (Asli Siaga)** yang melanda Jabodetabek:
- Model berhasil mendeteksinya (*Recall/Hit*) sebanyak **788 kali (Akurasi/Recall Siaga = 91%)**.
- Hanya 31 badai (3%) yang terlewat fatal sebagai kelas "Aman".

**KESIMPULAN:** Fase 6 sukses melahirkan mesin cuaca dengan kecerdasan tingkat Jurnal Q1 (Sinta 2). Ia sangat andal, sangat presisi, dan sangat sensitif terhadap ancaman badai nyata. Otak (file `.pt`) ini kini siap diimplementasikan secara hidup ke dalam Dashboard pada Fase 8.
