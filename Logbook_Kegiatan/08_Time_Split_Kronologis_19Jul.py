import os, matplotlib.pyplot as plt, numpy as np

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Pemotongan Data (Time-Split) Bebas Kebocoran")
# Buat data kronologis 2016 - 2026
years = np.arange(2016, 2026.1, 0.1)
data_volume = np.ones_like(years) * 100

plt.figure(figsize=(10, 3.5))
# Latih (2016 - 2023)
plt.fill_between(years, 0, data_volume, where=(years < 2024), color='forestgreen', alpha=0.7, label='Data Latih (70%) - 2016 s/d 2023')
# Validasi (2024)
plt.fill_between(years, 0, data_volume, where=(years >= 2024) & (years < 2025), color='gold', alpha=0.8, label='Data Validasi (15%) - 2024')
# Ujian (2025)
plt.fill_between(years, 0, data_volume, where=(years >= 2025), color='crimson', alpha=0.8, label='Data Ujian Blind-Test (15%) - 2025')

plt.title("Pemotongan Linimasa Absolut (Mencegah Kebocoran Autokorelasi Cuaca)", fontsize=12)
plt.xlabel("Tahun Observasi", fontsize=11)
plt.yticks([])
plt.xticks(np.arange(2016, 2027, 1))
plt.legend(loc='upper left')
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_08_Time_Split.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
