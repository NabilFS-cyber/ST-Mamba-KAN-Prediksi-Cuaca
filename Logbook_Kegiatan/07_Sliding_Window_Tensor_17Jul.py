import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(B1_PATH)
# Sortir berdasarkan waktu
df_stat = df[df['STATION'] == df['STATION'].iloc[0]].sort_values('DATE').head(25)

# Buat sliding window
windows = []
for i in range(10):
    windows.append(df_stat['tp'].values[i:i+14])
    
matrix = np.array(windows)

plt.figure(figsize=(12, 5))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Intensitas Cuaca (tp)'})
plt.title("Arsitektur Memori Sliding Window (14-Hari Look-back) Tensor", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Langkah Waktu Historis (T-13 sampai T-0)")
plt.ylabel("Sampel Array (Hari ke-i)")

out = os.path.join(VISUAL_DIR, "Hari_07_Sliding_Window_Tensor.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
