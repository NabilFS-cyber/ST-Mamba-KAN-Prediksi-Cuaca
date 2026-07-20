import os, numpy as np, matplotlib.pyplot as plt

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Sinkronisasi Zona Waktu (UTC vs WIB)")
jam_utc = np.arange(0, 24)
jam_wib = (jam_utc + 7) % 24
radiasi = np.exp(-0.1 * (jam_utc - 12)**2)

plt.figure(figsize=(10, 4))
plt.plot(jam_utc, radiasi, label="Radiasi (Skala UTC)", linestyle='--')
plt.plot(jam_wib, radiasi, label="Radiasi (Digeser ke WIB GMT+7)")
plt.title("Pergeseran Puncak Radiasi Matahari (Termodinamika)")
plt.xlabel("Jam (24H)")
plt.legend()
output_path = os.path.join(VISUAL_DIR, "Hari_10_Sinkronisasi_Zona_Waktu.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
