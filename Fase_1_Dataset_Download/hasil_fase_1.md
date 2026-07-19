# 🌐 HASIL FASE 1: DOWNLOAD DATASET ERA5-LAND (ECMWF) & AKUISISI BMKG

Fase 1 adalah titik nol dari pengembangan arsitektur ST-Mamba-KAN. Pada tahap ini, kita mengumpulkan data iklim masa lalu beresolusi tinggi langsung dari satelit Eropa dan menyandingkannya dengan data observasi stasiun darat Indonesia (BMKG).

---

## 📥 1. Sumber Data (Copernicus CDS)
- **Platform:** ECMWF Copernicus Climate Data Store (CDS)
- **API:** Diunduh secara otomatis menggunakan pustaka Python `cdsapi`
- **Dataset Utama:** `reanalysis-era5-land` (Dataset reanalisis cuaca permukaan)
- **Cakupan Temporal:** 2016 hingga Mei 2026 (Data historis satelit 10+ Tahun)
- **Resolusi Spasial:** Grid berukuran 0.1° × 0.1° (Area batas bujur lintang Jabodetabek)
- **Resolusi Temporal:** Observasi per jam (24 rekam data per hari)

---

## 🌡️ 2. Katalog Variabel Satelit (11 Fitur Cuaca ERA5)

Model membutuhkan pemahaman fisika atmosfer, oleh karenanya 11 fitur cuaca multidimensi diekstrak secara serentak:

| No | Nama Variabel Asli (API) | Keterangan |
|:--:|:---|:---|
| 1 | `10m_u_component_of_wind` | Komponen kecepatan angin horizontal (Timur/Barat) |
| 2 | `10m_v_component_of_wind` | Komponen kecepatan angin vertikal (Utara/Selatan) |
| 3 | `2m_dewpoint_temperature` | Suhu titik embun (indikator kelembapan) |
| 4 | `2m_temperature` | Suhu udara 2m di atas permukaan |
| 5 | `surface_pressure` | Tekanan udara permukaan darat |
| 6 | `total_precipitation` | Total presipitasi curah hujan satelit |
| 7 | `surface_net_solar_radiation` | Radiasi matahari netto di permukaan |
| 8 | `skin_temperature` | Suhu permukaan bumi (kulit bumi) |
| 9 | `volumetric_soil_water_layer_1` | Kandungan air dalam tanah lapisan 1 |
| 10| `volumetric_soil_water_layer_2` | Kandungan air dalam tanah lapisan 2 |
| 11| `evaporation_from_bare_soil` | Penguapan dari tanah kosong |

---

## 🌧️ 3. Katalog Variabel Darat (6 Fitur Observasi BMKG)

Sebagai pasangan kalibrasi (Ground Truth), data dari stasiun observasi darat BMKG (Periode Juni 2024 - Mei 2026) digunakan. Terdapat 6 fitur lapangan:

| No | Nama Variabel | Keterangan |
|:--:|:---|:---|
| 1 | `TX` | Suhu Maksimum Harian (°C) |
| 2 | `RH_AVG` | Kelembapan Udara Rata-rata (%) |
| 3 | `RR` | Curah Hujan Harian (mm) - **(Label Target Utama)** |
| 4 | `SS` | Lama Penyinaran Matahari (Jam) |
| 5 | `FF_X` | Kecepatan Angin Maksimum (m/s atau knot) |
| 6 | `DDD_X` | Arah Angin Maksimum (Derajat) |

---

## 📦 4. Output Data

- **Nama File Akhir:** Ratusan file bulanan dengan format `dataset_era5_land_jabodetabek_YYYY_MM.nc` (mulai dari 2016_01 hingga 2026_05).
- **Penyimpanan:** File ini disimpan secara permanen di direktori `/content/drive/MyDrive/Riset_ERA5_Land/` agar dapat diakses tanpa harus mengunduh ulang.

File data mentah dari satelit dan stasiun darat ini nantinya akan diproses lebih lanjut oleh **Fase 2 (Quality Check)** dan **Fase 3 (Fusi Data)**.
