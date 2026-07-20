import os, pandas as pd, matplotlib.pyplot as plt
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
B2_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas2_finetune.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df1 = pd.read_parquet(B1_PATH)
df2 = pd.read_parquet(B2_PATH)

df1['DATE'] = pd.to_datetime(df1['DATE'])
df2['DATE'] = pd.to_datetime(df2['DATE'])

plt.figure(figsize=(12, 4))
plt.plot([df1['DATE'].min(), df1['DATE'].max()], [1, 1], linewidth=30, color='royalblue', label='Brankas 1 (Pre-Training Masa Lalu)')
plt.plot([df2['DATE'].min(), df2['DATE'].max()], [1.2, 1.2], linewidth=30, color='darkorange', label='Brankas 2 (Fine-Tuning & Blind Test)')

plt.title("Pemotongan Waktu Kronologis (Mencegah Data Leakage)", fontsize=14, fontweight='bold', pad=15)
plt.yticks([])
plt.xlabel("Tahun")
plt.ylim(0.5, 1.7)
plt.legend(loc='lower center', ncol=2)
plt.grid(True, alpha=0.3, axis='x')

out = os.path.join(VISUAL_DIR, "Hari_08_Time_Split_Kronologis.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
