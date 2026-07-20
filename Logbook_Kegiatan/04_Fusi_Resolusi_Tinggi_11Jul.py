import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Fusi: ERA5 tp (m) vs BMKG RR (mm)
if 'tp' in df.columns and 'RR' in df.columns:
    df['tp_mm'] = df['tp'] * 1000

    plt.figure(figsize=(8, 8))
    sns.regplot(data=df, x='tp_mm', y='RR', scatter_kws={'alpha':0.3, 'color':'royalblue'}, line_kws={'color':'red'})
    plt.title("Validasi Fusi Resolusi Tinggi: Satelit ERA5-Land vs Stasiun BMKG", fontsize=14, fontweight='bold')
    plt.xlabel("Curah Hujan Prediksi Satelit ERA5-Land (mm)")
    plt.ylabel("Curah Hujan Aktual Permukaan BMKG (mm)")
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
    plt.grid(True, alpha=0.3)
else:
    print("Kolom tidak ditemukan untuk perbandingan.")

out = os.path.join(VISUAL_DIR, "Hari_04_Fusi_Resolusi.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
