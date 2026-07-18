# =====================================================================
# FASE 1: PENGUNDUHAN & FUSI DATASET ERA5-LAND (2016 - 2026)
# =====================================================================
# Deskripsi:
# Script ini mengunduh dataset iklim dari Copernicus (ERA5-Land) 
# mencakup periode 10 tahun (2016 - Mei 2026).
# Fitur Utama:
# - Dual-Sub-Request untuk menghindari limit server Copernicus.
# - Auto-Unpack ZIP jika server membungkus paksa file NetCDF.
# - Penggabungan (Merge) menggunakan xarray engine netcdf4.
# =====================================================================

import os
import zipfile
import shutil
import cdsapi
import xarray as xr
from google.colab import drive

# ---------------------------------------------------------
# 1. KONFIGURASI DRIVE & CDS API
# ---------------------------------------------------------
drive.mount('/content/drive', force_remount=True)
DRIVE_OUT = '/content/drive/MyDrive/Riset_ERA5_Land/'
os.makedirs(DRIVE_OUT, exist_ok=True)

# Masukkan API Key Copernicus (Tanpa UID/Titik Dua)
api_key_baru = "aa22b52b-43ae-4bd9-81cc-79fd077582d9"
with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
    f.write("url: https://cds.climate.copernicus.eu/api\n")
    f.write(f"key: {api_key_baru.strip()}\n")

client = cdsapi.Client()
print("✅ Konfigurasi API dan Google Drive Berhasil!")

# ---------------------------------------------------------
# 2. DEFINISI VARIABEL DAN RENTANG WAKTU
# ---------------------------------------------------------
dataset = "reanalysis-era5-land"
vars_group_A = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_dewpoint_temperature", "2m_temperature", "surface_pressure"
]
vars_group_B = [
    "total_precipitation", "surface_net_solar_radiation", "skin_temperature",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
    "evaporation_from_bare_soil"
]

times_all = [f"{str(h).zfill(2)}:00" for h in range(24)]

# ---------------------------------------------------------
# 3. FUNGSI PEMBONGKAR ZIP (ANTI CORRUPT)
# ---------------------------------------------------------
def ekstrak_dan_baca_netcdf(file_path):
    """
    Mengecek apakah file dienkapsulasi ZIP oleh Copernicus.
    Jika ZIP, ekstrak semua isinya dan gabungkan internal.
    Jika murni NetCDF, langsung kembalikan dataset xarray.
    """
    if zipfile.is_zipfile(file_path):
        temp_dir = "temp_unpack_" + os.path.basename(file_path).split('.')[0]
        os.makedirs(temp_dir, exist_ok=True)
        
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(temp_dir)
            
        # Gabungkan semua pecahan .nc di dalam zip
        pecahan = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.nc')]
        ds_list = [xr.open_dataset(f, engine="netcdf4") for f in pecahan]
        ds_merged = xr.merge(ds_list, compat="override")
        
        # Bersihkan pecahan
        shutil.rmtree(temp_dir)
        return ds_merged, ds_list
    else:
        # File sudah murni NetCDF
        ds = xr.open_dataset(file_path, engine="netcdf4")
        return ds, []

# ---------------------------------------------------------
# 4. LOOPING EKSEKUSI UTAMA (2016 - 2026)
# ---------------------------------------------------------
years = [str(y) for y in range(2016, 2027)]

for year in years:
    # Tahun 2026 hanya tersedia hingga Mei
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    if year == "2026":
        months = ["01", "02", "03", "04", "05"]
        
    for month in months:
        filename_drive = os.path.join(DRIVE_OUT, f"dataset_era5_land_jabodetabek_{year}_{month}.nc")
        if os.path.exists(filename_drive):
            print(f"ℹ️ {year}-{month} sudah aman di Drive. Melewati...")
            continue
            
        # Menghitung jumlah hari dinamis (Kabisat/Bulan)
        if month == "02":
            is_kabisat = (int(year) % 4 == 0) and (int(year) % 100 != 0 or int(year) % 400 == 0)
            days_target = [f"{d:02d}" for d in range(1, 30 if is_kabisat else 29)]
        elif month in ["04", "06", "09", "11"]:
            days_target = [f"{d:02d}" for d in range(1, 31)]
        else:
            days_target = [f"{d:02d}" for d in range(1, 32)]

        print(f"\n⏳ Memproses {year}-{month} (Total Hari: {len(days_target)})...")
        
        base_request = {
            "year": [year], "month": [month],
            "day": days_target, "time": times_all,
            "area": [-5.9, 106.3, -6.8, 107.3], # Bounding Box Jabodetabek
            "data_format": "netcdf",
            "download_format": "unarchived"
        }

        file_A = f"temp_A_{year}_{month}.nc"
        file_B = f"temp_B_{year}_{month}.nc"

        try:
            # Unduh Porsi A
            req_A = base_request.copy(); req_A["variable"] = vars_group_A
            client.retrieve(dataset, req_A, file_A)

            # Unduh Porsi B
            req_B = base_request.copy(); req_B["variable"] = vars_group_B
            client.retrieve(dataset, req_B, file_B)

            print("   ---> Membongkar dan Menyatukan Porsi A & B...")
            ds_A, clean_A = ekstrak_dan_baca_netcdf(file_A)
            ds_B, clean_B = ekstrak_dan_baca_netcdf(file_B)
            
            # Fusi Master
            ds_final = xr.merge([ds_A, ds_B], compat="override")
            ds_final.to_netcdf(filename_drive, engine="netcdf4")
            
            print(f"✅ SUKSES! 11 Variabel untuk {year}-{month} Terkunci di Drive.")

            # Pembersihan Memori
            ds_final.close(); ds_A.close(); ds_B.close()
            for d in clean_A + clean_B: d.close()
            if os.path.exists(file_A): os.remove(file_A)
            if os.path.exists(file_B): os.remove(file_B)

        except Exception as e:
            print(f"❌ Gagal memproses {year}-{month}. Kendala: {e}")
            if os.path.exists(file_A): os.remove(file_A)
            if os.path.exists(file_B): os.remove(file_B)

print("\n🎉 SELURUH DATA ERA5-LAND (2016-2026) SUKSES DIUNDUH DAN DIBERSIHKAN!")
