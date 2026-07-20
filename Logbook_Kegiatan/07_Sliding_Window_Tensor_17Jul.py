import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
if not os.path.exists(B1_PATH):
    B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv"

os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(B1_PATH) if B1_PATH.endswith('.parquet') else pd.read_csv(B1_PATH)

station_col = next((c for c in ['Nama_Stasiun', 'STATION', 'stasiun', 'Station'] if c in df.columns), None)
val_col = next((c for c in ['tp', 'RR', 'rr', 't2m', 'TX'] if c in df.columns), df.select_dtypes(include=[np.number]).columns[0])

if station_col:
    df_stat = df[df[station_col] == df[station_col].iloc[0]].head(30)
else:
    df_stat = df.head(30)

series = df_stat[val_col].values
windows = [series[i:i+14] for i in range(min(10, len(series)-14))]
matrix = np.array(windows)

plt.figure(figsize=(12, 5))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': f'Intensitas {val_col}'})
plt.title(f"Arsitektur Memori Sliding Window (14-Hari Look-back) Tensor [{val_col}]", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Langkah Waktu Historis (T-13 sampai T-0)")
plt.ylabel("Sampel Array (Hari ke-i)")

out = os.path.join(VISUAL_DIR, "Hari_07_Sliding_Window_Tensor.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
