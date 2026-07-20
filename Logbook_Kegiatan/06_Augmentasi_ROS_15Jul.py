import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(B1_PATH)
df['CLASS'] = (df['RR'] >= 50).astype(int)

X = df[['tp']].values # dummy X untuk resampling indeks
y = df['CLASS'].values

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y, ax=axes[0], palette=['lightblue', 'red'])
axes[0].set_title("Sebelum Augmentasi (Timpa Mayoritas Aman)")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Aman (<50mm)', 'Siaga (>=50mm)'])
axes[0].set_ylabel('Jumlah Hari')

sns.countplot(x=y_res, ax=axes[1], palette=['lightblue', 'red'])
axes[1].set_title("Sesudah Random Over-Sampling (Porsi Seimbang)")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Aman (<50mm)', 'Siaga (>=50mm)'])
axes[1].set_ylabel('')

plt.suptitle("Penanganan Imbalance Data Hujan Ekstrem Menggunakan ROS", fontsize=14, fontweight='bold')

out = os.path.join(VISUAL_DIR, "Hari_06_Augmentasi_ROS.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
