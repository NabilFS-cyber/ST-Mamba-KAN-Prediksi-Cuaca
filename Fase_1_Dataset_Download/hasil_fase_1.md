# 🌐 HASIL FASE 1: DOWNLOAD DATASET ERA5-LAND (ECMWF)

Fase 1 adalah titik nol dari pengembangan arsitektur ST-Mamba-KAN. Pada tahap ini, kita mengumpulkan data iklim masa lalu beresolusi tinggi langsung dari satelit Eropa untuk memodelkan sistem hidrometeorologi di atas wilayah Jabodetabek.

---

## 📥 1. Sumber Data (Copernicus CDS)
- **Platform:** ECMWF Copernicus Climate Data Store (CDS)
- **API:** Diunduh secara otomatis menggunakan pustaka Python `cdsapi`
- **Dataset Utama:** `reanalysis-era5-land` (Dataset reanalisis cuaca permukaan paling komprehensif saat ini)
- **Cakupan Temporal:** 1 Januari 2005 hingga 31 Desember 2024 (Data historis 20 Tahun Penuh)
- **Resolusi Spasial:** Grid berukuran 0.1° × 0.1° (Area batas bujur lintang Jabodetabek: -6.0° s/d -7.0°S, dan 106.6° s/d 107.0°E)
- **Resolusi Temporal:** 4 observasi per hari (Pukul 00:00, 06:00, 12:00, dan 18:00 UTC)

---

## 🌡️ 2. Katalog Variabel (18 Fitur Cuaca)

Model membutuhkan pemahaman fisika atmosfer yang lengkap, oleh karenanya 18 fitur cuaca multidimensi diekstrak secara serentak. Fitur ini terdiri dari suhu, kelembapan, radiasi, hingga evaporasi:

| No | Nama Variabel Asli (API) | Singkatan | Satuan / Deskripsi |
|:--:|:---|:---:|:---|
| 1 | `2m_temperature` | **t2m** | Kelvin (K) - Suhu udara 2m di atas daratan |
| 2 | `2m_dewpoint_temperature` | **d2m** | Kelvin (K) - Suhu titik embun (indikator kelembapan) |
| 3 | `total_precipitation` | **tp** | Meter (m) - Total presipitasi curah hujan satelit |
| 4 | `surface_pressure` | **sp** | Pascal (Pa) - Tekanan udara permukaan darat |
| 5 | `10m_u_component_of_wind` | **u10** | m/s - Komponen kecepatan angin horizontal (Timur/Barat) |
| 6 | `10m_v_component_of_wind` | **v10** | m/s - Komponen kecepatan angin vertikal (Utara/Selatan) |
| 7 | `total_cloud_cover` | **tcc** | (0-1) - Kerapatan awan (tutupan awan total) |
| 8 | `soil_temperature_level_1` | **stl1** | Kelvin (K) - Suhu tanah lapisan teratas |
| 9 | `volumetric_soil_water_layer_1`| **swvl1** | m³/m³ - Kandungan air dalam tanah |
| 10| `surface_latent_heat_flux` | **slhf** | J/m² - Fluks panas laten (akumulatif) |
| 11| `surface_sensible_heat_flux` | **sshf** | J/m² - Fluks panas peka (akumulatif) |
| 12| `surface_net_solar_radiation` | **ssr** | J/m² - Radiasi matahari netto di permukaan |
| 13| `surface_net_thermal_radiation`| **str** | J/m² - Radiasi termal netto di permukaan |
| 14| `evaporation` | **e** | m hujan ekuivalen - Tingkat penguapan |
| 15| `convective_precipitation` | **cp** | m - Presipitasi konvektif (hujan badai lokal) |
| 16| `total_column_water_vapour` | **tcwv** | kg/m² - Total uap air dalam kolom atmosfer |
| 17| `leaf_area_index_high_vegetation`| **lai_hv**| m²/m² - Indeks kerapatan daun pohon tinggi |
| 18| `leaf_area_index_low_vegetation` | **lai_lv**| m²/m² - Indeks kerapatan daun rumput/rendah |

---

## 📦 3. Output Data

- **Nama File Akhir:** `ERA5_Jabodetabek_2005_2024.nc` (Berekstensi NetCDF4, standar global klimatologi).
- **Ukuran File:** ~1.5 GB
- **Penyimpanan:** File ini disimpan secara permanen di direktori `/content/drive/MyDrive/Riset_ERA5_Land/` agar dapat diakses seumur hidup tanpa harus meminta data ulang ke API Eropa.

File `*.nc` ini nantinya akan diproses lebih lanjut oleh **Fase 2 (Quality Check)** dan **Fase 3 (Fusi Data Stasiun BMKG)** untuk membersihkan data mentah dari satelit.
