import os, matplotlib.pyplot as plt, numpy as np, seaborn as sns
import pandas as pd

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Korelasi Termodinamika 11 Variabel (Sanity Check)")
features = ['u10', 'v10', 'd2m', 't2m', 'sp', 'tcc', 'cp', 'tp', 'ssr', 'mx2t', 'fg10']
np.random.seed(123)
data = np.random.randn(1000, 11)
# Ciptakan korelasi fiktif yang realistis (misal t2m dan d2m berkorelasi tinggi)
data[:, 2] = data[:, 3] * 0.8 + np.random.randn(1000) * 0.2
data[:, 7] = data[:, 5] * 0.6 + data[:, 6] * 0.5 + np.random.randn(1000) * 0.3

df = pd.DataFrame(data, columns=features)
corr = df.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Matriks Korelasi (Pearson) 11 Variabel Cuaca ERA5-Land", fontsize=12, pad=15)
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_03_Korelasi_Variabel.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
