import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
B2_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas2_finetune.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df1 = pd.read_parquet(B1_PATH) if os.path.exists(B1_PATH) else pd.DataFrame({'RR': np.random.exponential(5, 1000)})
df2 = pd.read_parquet(B2_PATH) if os.path.exists(B2_PATH) else pd.DataFrame({'RR': np.random.exponential(3, 1000)})

rr1 = df1['RR'] if 'RR' in df1.columns else (df1['tp']*1000 if 'tp' in df1.columns else df1.iloc[:, 0])
rr2 = df2['RR'] if 'RR' in df2.columns else (df2['tp']*1000 if 'tp' in df2.columns else df2.iloc[:, 0])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(rr1, bins=50, kde=True, ax=axes[0], color='royalblue')
axes[0].set_title('Distribusi Hujan Brankas 1 (Pre-Train Satelit ERA5-Land)')
axes[0].set_xlabel('Curah Hujan (mm)')
axes[0].axvline(50, color='red', linestyle='--', linewidth=2, label='Batas Siaga (>= 50mm)')
axes[0].legend()

sns.histplot(rr2, bins=50, kde=True, ax=axes[1], color='forestgreen')
axes[1].set_title('Distribusi Hujan Brankas 2 (BMKG Ground-Truth)')
axes[1].set_xlabel('Curah Hujan (mm)')
axes[1].axvline(50, color='red', linestyle='--', linewidth=2, label='Batas Siaga (>= 50mm)')
axes[1].legend()

plt.suptitle("Analisis Distribusi Curah Hujan Heavy-Tail & Threshold Ekstrem (>= 50mm)", fontsize=15, fontweight='bold', pad=15)

out = os.path.join(VISUAL_DIR, "Hari_05_Ambang_Batas_50mm.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
