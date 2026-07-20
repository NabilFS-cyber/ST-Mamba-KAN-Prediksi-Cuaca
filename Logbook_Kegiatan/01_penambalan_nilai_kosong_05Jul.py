import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)
BMKG_FILE = "/content/drive/MyDrive/Riset_ERA5_Land/Data_BMKG/Stasiun Meteorologi Kemayoran.xlsx"

print("Simulasi Penambalan Nilai Kosong BMKG")
if os.path.exists(BMKG_FILE):
    df = pd.read_excel(BMKG_FILE, skiprows=8)
    col = 'Tavg' if 'Tavg' in df.columns else df.columns[2]
    
    data_asli = df[col].iloc[0:100].copy()
    data_mentah = data_asli.copy()
    np.random.seed(42)
    missing = np.random.choice(100, 15, replace=False)
    data_mentah.iloc[missing] = np.nan
    data_tambal = data_mentah.ffill().bfill()
    
    plt.figure(figsize=(10, 4))
    plt.plot(data_tambal, label="Ditambal (ffill/bfill)", color='green', alpha=0.7)
    plt.plot(data_mentah, label="Mentah (Bolong)", color='red', marker='o', linestyle='dashed', linewidth=1)
    plt.title("Perbaikan Integritas Data BMKG")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(VISUAL_DIR, "Penambalan_Data.png")
    plt.savefig(output_path)
    print("-> Visualisasi disimpan di", output_path)
else:
    print(f"File BMKG tidak ditemukan di Google Drive: {BMKG_FILE}")
