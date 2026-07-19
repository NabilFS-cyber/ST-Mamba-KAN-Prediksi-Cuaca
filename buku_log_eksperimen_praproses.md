# 📖 BUKU LOG EKSPERIMEN: PRAPROSES & REKAYASA DATA (DATA ENGINEERING)

Dokumen ini adalah jurnal ilmiah yang mencatat seluruh jejak rekam uji coba (*trial and error*) dalam proses pembersihan, fusi, dan augmentasi data. Pemisahan log ini bertujuan untuk membuktikan secara empiris tingginya kompleksitas dan kedalaman analisis data (*Data Engineering*) sebelum masuk ke tahap pemodelan AI.

---

## 🧪 DAFTAR EKSPERIMEN PRAPROSES DATA

| No. Eksperimen | Tanggal | Deskripsi / Hipotesis Praproses | Parameter yang Diubah | Hasil & Metrik Evaluasi | Kesimpulan / Tindakan Selanjutnya |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PP-001** | *[Tanggal]* | *Contoh: Uji coba interpolasi spasial satelit ke stasiun.* | *Metode Nearest Neighbor vs Bilinear* | *Bilinear lebih halus dan RMSE error cuaca berkurang 12%* | *Gunakan Bilinear Interpolation di Fase 4.* |
| **PP-002** | *[Tanggal]* | *Contoh: Penanganan Class Imbalance badai ekstrem.* | *SMOTE 2D biasa vs Flattened SMOTETomek* | *SMOTE 2D merusak label regresi. Flattened berhasil menjaga nilai milimeter aslinya.* | *Tetapkan Flattened SMOTETomek di Fase 5.* |
| **PP-003** | ... | ... | ... | ... | ... |
| **PP-004** | ... | ... | ... | ... | ... |
| **PP-005** | ... | ... | ... | ... | ... |

*(Silakan tambahkan baris tabel sesuai dengan puluhan percobaan praproses yang telah Anda lakukan).*

---

## 📌 CATATAN PENTING & DISKUSI PENEMUAN
*Bagian ini dapat diisi dengan temuan-temuan unik (insight) selama merekayasa data, seperti anomali sensor pada stasiun tertentu, efek dari menggeser zona waktu (GMT ke WIB), atau keputusan memotong data menjadi Dual Brankas.*
