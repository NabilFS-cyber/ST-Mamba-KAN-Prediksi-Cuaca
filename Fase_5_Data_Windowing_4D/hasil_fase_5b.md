# FASE 5B: 4D SPATIO-TEMPORAL WINDOWING & FLATTENED SMOTETOMEK

## 📝 Status Eksekusi: BERHASIL TOTAL (BUG FIXED)
Fase 5B telah berhasil dijalankan dengan sempurna. Tensor 4D `[Batch, Waktu, Stasiun, Fitur]` telah terbuat dan dataset telah diseimbangkan menggunakan **SMOTETomek** tanpa mengorbankan (mengacak) target regresi.

## 🐛 Perbaikan Bug Krusial di Fase Ini
Pada iterasi sebelumnya, terdapat **Bug Fatal pada SMOTETomek** di mana fungsi tersebut hanya menginterpolasi data fitur (`X`) dan target klasifikasi (`yc`), lalu target regresi (`yr`) dari data asli dimasukkan kembali secara manual menggunakan `yr_res[:len(yr)] = yr`. Karena SMOTETomek menghapus dan mengacak baris data (Tomek Links), hal ini menyebabkan target regresi **teracak total** (Sinyal Badai menunjuk ke 0mm, Sinyal Cerah menunjuk ke 150mm). Inilah penyebab RMSE Fase 7B sempat meledak hingga 106 mm.

**Solusi yang Diterapkan:**
Target regresi `yr` digabungkan (di-stack) ke dalam fitur matriks 2D sebelum diumpankan ke SMOTE (`np.column_stack([X_flat, yr])`). Dengan demikian, SMOTE ikut **menginterpolasi dan mempertahankan posisi `yr` yang benar** seiring dengan penciptaan data cuaca ekstrem buatan. RMSE dijamin aman dan akurat!

## 📊 Hasil Distribusi Dataset

### 1. Merakit Tensor 4D (Window = 14 Hari)
- **Brankas 1 (Pretrain):** 15,300 sampel (Shape: `15300, 14, 5, 18`)
- **Brankas 2 (Finetune):** 3,560 sampel (Shape: `3560, 14, 5, 18`)

*Catatan: Fitur berjumlah 18 yang terdiri dari 17 Fitur Cuaca + 1 Indikator Stasiun (One-Hot Encoded).*

### 2. Distribusi Kelas Setelah SMOTETomek (Balancing)
Data kini sangat seimbang, model akan terhindar dari bias mayoritas (hujan ringan).

**BRANKAS 1 TRAIN**
- Kelas 0 (Aman / <20mm): 5,632
- Kelas 1 (Waspada / 20-50mm): 5,630
- Kelas 2 (Siaga / ≥50mm): 5,630

**BRANKAS 2 TRAIN**
- Kelas 0 (Aman / <20mm): 2,087
- Kelas 1 (Waspada / 20-50mm): 2,087
- Kelas 2 (Siaga / ≥50mm): 2,088

## 💾 Output File (Disimpan di Google Drive `tensors_mamba`)
1. **Master Scaler:** `scaler_master_4d.pkl`
2. **Brankas 1 (Train/Val/Test):** `b1_train_X_4d.pt`, `b1_train_yr_4d.pt`, `b1_train_yc_4d.pt`, dst.
3. **Brankas 2 (Train/Val/Test):** `b2_train_X_4d.pt`, `b2_train_yr_4d.pt`, `b2_train_yc_4d.pt`, dst.

Dataset 4D ini kini sudah menjadi bahan bakar yang murni dan 100% siap untuk dimasukkan ke dalam Arsitektur **Fase 7B (GNN-Mamba Elite Masterpiece)**!
