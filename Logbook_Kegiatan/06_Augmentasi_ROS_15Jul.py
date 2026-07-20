import os, matplotlib.pyplot as plt, numpy as np

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Koreksi Ketimpangan Kelas via Random Over-Sampling (ROS)")
labels = ['Normal (<50mm)', 'Siaga (>=50mm)']
sblm_ros = [360000, 5632]
stlh_ros = [360000, 360000]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, sblm_ros, width, label='Sebelum ROS (Imbalance Parah)', color='#1f77b4')
rects2 = ax.bar(x + width/2, stlh_ros, width, label='Pasca ROS (Data Setara)', color='#ff7f0e')

ax.set_ylabel('Jumlah Sampel (Baris Data)', fontsize=11)
ax.set_title('Penyetaraan Distribusi Kelas Ekstrem untuk Menghindari Overfitting AI', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Format y-axis to text
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_06_Augmentasi_ROS.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
