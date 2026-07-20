import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"

os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH) if DATA_PATH.endswith('.csv') else pd.read_parquet(DATA_PATH)

x_col = next((c for c in ['tp', 'tp_mm', 'ERA5_tp'] if c in df.columns), None)
y_col = next((c for c in ['RR', 'rr', 'RAINFALL', 'rainfall_bmkg'] if c in df.columns), None)

plt.figure(figsize=(8, 8))
if x_col and y_col:
    # Selalu kalikan 1000 jika kolom adalah 'tp' (ERA5 m ke mm) atau rata-ratanya sangat kecil
    x_val = df[x_col] * 1000.0 if (df[x_col].mean() < 0.1 or x_col == 'tp') else df[x_col]
    y_val = df[y_col]
    sns.regplot(x=x_val, y=y_val, scatter_kws={'alpha':0.3, 'color':'royalblue'}, line_kws={'color':'red'})
    plt.xlabel(f"Curah Hujan Satelit ERA5-Land ({x_col}) [mm]")
    plt.ylabel(f"Curah Hujan Stasiun BMKG ({y_col}) [mm]")
    plt.xlim(-5, 100)
    plt.ylim(-5, 100)
else:
    num_cols = df.select_dtypes(include=[np.number]).columns
    sns.regplot(x=df[num_cols[0]], y=df[num_cols[1]], scatter_kws={'alpha':0.3, 'color':'royalblue'}, line_kws={'color':'red'})

plt.title("Validasi Fusi Resolusi Tinggi: Satelit ERA5-Land vs Stasiun BMKG", fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3)

out = os.path.join(VISUAL_DIR, "Hari_04_Fusi_Resolusi.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
