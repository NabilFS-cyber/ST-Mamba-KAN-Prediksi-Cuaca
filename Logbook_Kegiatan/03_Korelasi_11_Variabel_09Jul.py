import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

# Pilih kolom cuaca numerik
all_cols = ['t2m', 'tp', 'RH_AVG', 'RR', 'SS', 'TX', 'sp', 'swvl1', 'evabs', 'FF_X', 'd2m']
cols_real = [c for c in all_cols if c in df.columns]

if not cols_real:
    cols_real = [c for c in df.columns if df[c].dtype in ['float32', 'float64', 'int64']][:11]
    
corr = df[cols_real].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=.5)
plt.title("Matriks Korelasi Pearson 11 Variabel Cuaca (Sanity Check Logika Alam)", fontsize=14, fontweight='bold', pad=15)

out = os.path.join(VISUAL_DIR, "Hari_03_Korelasi_Variabel.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
