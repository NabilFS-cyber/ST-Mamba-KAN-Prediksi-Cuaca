# Laporan Eksekusi: Fase 1 (Pengunduhan & Eksplorasi Dataset)

Fase 1 berfokus pada pengumpulan, pengunduhan, dan pemahaman karakteristik dua sumber data utama yang akan digunakan untuk melatih Model AI Peringatan Dini Banjir Jabodetabek. Data ini terdiri dari **Data Satelit (ERA5-Land)** dan **Data Observasi Darat (BMKG)**.

---

## 🛰️ 1. DATA SATELIT: Copernicus ERA5-Land (Resolusi Per Jam)
Data ini bertindak sebagai **"Fitur Cuaca & Iklim" (Inputs)** untuk melatih pemahaman fisika AI.

### 📌 Karakteristik Pengunduhan:
- **Periode Waktu**: 1 Januari 2016 – 31 Mei 2026 (10+ Tahun).
- **Resolusi Temporal**: **PER JAM (Hourly / 24 Jam)**. *Ini sangat penting karena cuaca ekstrem bisa terjadi dan berlalu hanya dalam hitungan jam.*
- **Resolusi Spasial**: Mencakup *Bounding Box* area Jabodetabek (Latitude: -5.9 hingga -6.8, Longitude: 106.3 hingga 107.3).

### 📊 11 Variabel Meteorologi yang Diamankan:
Data dipecah menjadi dua porsi (A dan B) untuk menghindari *timeout* dari server Copernicus, lalu dijahit secara presisi menjadi satu file `.nc` per bulan:
1. `10m_u_component_of_wind` (Kecepatan Angin U)
2. `10m_v_component_of_wind` (Kecepatan Angin V)
3. `2m_dewpoint_temperature` (Suhu Titik Embun)
4. `2m_temperature` (Suhu Udara)
5. `surface_pressure` (Tekanan Permukaan)
6. `total_precipitation` (Curah Hujan Total - *Parameter Paling Krusial*)
7. `surface_net_solar_radiation` (Radiasi Matahari Neto)
8. `skin_temperature` (Suhu Permukaan Tanah)
9. `volumetric_soil_water_layer_1` (Kandungan Air Tanah Lapisan 1)
10. `volumetric_soil_water_layer_2` (Kandungan Air Tanah Lapisan 2)
11. `evaporation_from_bare_soil` (Evaporasi dari Tanah Kosong)

---

## 🏢 2. DATA OBSERVASI DARAT: Stasiun BMKG (Ground-Truth)
Data ini bertindak sebagai **"Kunci Jawaban" (Targets/Labels)** untuk mengoreksi tebakan (prediksi) AI. Data BMKG ini berbasis harian (Daily).

Berdasarkan eksplorasi folder `C:\kuliah nabil\DLL\PKM\PENDANAAN\DataSet\BMKG FIX PERSTASIUN PERFILE`, kita telah mengamankan 5 file dataset Excel (kunci jawaban murni) dari 5 stasiun pengamatan iklim yang tersebar secara strategis, yaitu:

1. 📍 **Stasiun Meteorologi Soekarno Hatta** *(Mewakili area Barat/Tangerang)*
2. 📍 **Stasiun Meteorologi Maritim Tanjung Priok** *(Mewakili area Utara/Pesisir)*
3. 📍 **Stasiun Meteorologi Kemayoran** *(Mewakili area Pusat Jakarta)*
4. 📍 **Stasiun Meteorologi Citeko** *(Mewakili area Selatan/Dataran Tinggi Puncak)*
5. 📍 **Stasiun Klimatologi Jawa Barat** *(Mewakili area Bogor/Hulu)*

### 📌 Karakteristik & Rentang Waktu Data BMKG:
- **Periode Waktu**: 5 Juni 2024 hingga 31 Mei 2026.
- **Resolusi Temporal**: Harian (*Daily*).
- Kelima stasiun ini nantinya menjadi titik referensi tata ruang untuk membangun matriks **GNN (Graph Neural Network)**.

### 📊 Variabel BMKG yang Digunakan:
Berdasarkan data mentah Excel yang diunduh dari pusat data BMKG, terdapat 6 parameter cuaca darat yang terekam, yaitu:
1. **TX**: Suhu Udara Maksimum (°C)
2. **RH_AVG**: Kelembapan Udara Rata-rata (%)
3. **RR**: Curah Hujan (mm) ➔ *Ini adalah Target Utama (Ground-Truth/Label) model AI.*
4. **SS**: Lamanya Penyinaran Matahari (jam)
5. **FF_X**: Kecepatan Angin Maksimum (m/s atau knot)
6. **DDD_X**: Arah Angin saat kecepatan maksimum (derajat)

Nilai **RR (Curah Hujan)** inilah yang menjadi patokan mutlak untuk mengkategorikan peringatan banjir: **Aman (<20mm)**, **Waspada (20-50mm)**, atau **Siaga (≥50mm)**.

---

## ✅ Kesimpulan Fase 1
Kedua pilar utama data (*Input* ERA5-Land Per Jam dan *Target* BMKG Harian dari 5 Stasiun) telah berhasil diamankan sepenuhnya. 

Langkah selanjutnya (Fase 2) adalah memproses dan menyamakan (*align*) zona waktu serta dimensi kedua data ini, mengingat ERA5-Land beresolusi *Per Jam*, sedangkan data observasi BMKG beresolusi *Harian*.
