import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

if os.path.exists(B1_PATH):
    df = pd.read_parquet(B1_PATH)
    rr = df['RR'] if 'RR' in df.columns else (df['tp']*1000 if 'tp' in df.columns else df.iloc[:,0])
    y = (rr >= 50).astype(int).values
else:
    y = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

X = y.reshape(-1, 1) # dummy X

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y, ax=axes[0], palette=['lightblue', 'red'])
axes[0].set_title("Sebelum Augmentasi (Sangat Timpang)")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Aman (<50mm)', 'Siaga (>=50mm)'])
axes[0].set_ylabel('Jumlah Hari')

sns.countplot(x=y_res, ax=axes[1], palette=['lightblue', 'red'])
axes[1].set_title("Sesudah Random Over-Sampling (ROS Seimbang)")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Aman (<50mm)', 'Siaga (>=50mm)'])
axes[1].set_ylabel('')

plt.suptitle("Penanganan Imbalance Data Hujan Ekstrem Menggunakan ROS", fontsize=14, fontweight='bold', y=0.98)

out = os.path.join(VISUAL_DIR, "Hari_06_Augmentasi_ROS.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
