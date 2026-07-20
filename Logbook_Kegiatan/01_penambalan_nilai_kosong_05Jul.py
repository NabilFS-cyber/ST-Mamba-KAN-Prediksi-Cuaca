import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)
BMKG_FILE = r"C:\kuliah nabil\DLL\PKM\PENDANAAN\Perancangan_Model_AI\Fase_1_Dataset_Download\Dataset\BMKG\Stasiun Meteorologi Kemayoran.xlsx"

print("[HARI 1] Simulasi Penambalan Nilai Kosong BMKG")
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
    plt.title("Hari 1: Perbaikan Integritas Data BMKG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "Hari_01_Penambalan_Data.png"))
    print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_01_Penambalan_Data.png"))
else:
    print("File BMKG tidak ditemukan untuk simulasi.")
