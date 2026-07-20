import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Pembersihan Missing Values (Ffill/Bfill)")
# Simulasi Data Berlubang (30 hari x 5 stasiun)
np.random.seed(42)
mentah = np.random.rand(5, 30)
mask = np.random.choice([True, False], size=(5, 30), p=[0.15, 0.85])
mentah[mask] = np.nan

df = pd.DataFrame(mentah)
bersih = df.ffill(axis=1).bfill(axis=1)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
sns.heatmap(df.isnull(), ax=axes[0], cmap='Reds', cbar=False)
axes[0].set_title("Data Mentah BMKG (Garis Merah = Data Hilang/Missing)", fontsize=11)
axes[0].set_ylabel("ID Stasiun")

sns.heatmap(bersih.isnull(), ax=axes[1], cmap='Greens', cbar=False)
axes[1].set_title("Pasca Imputasi Forward/Backward Fill (100% Utuh)", fontsize=11)
axes[1].set_ylabel("ID Stasiun")
axes[1].set_xlabel("Indeks Waktu Harian")

plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_02_Missing_Values.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
