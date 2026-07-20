import os, numpy as np, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 10] Sinkronisasi Zona Waktu (UTC vs WIB)")
jam_utc = np.arange(0, 24)
jam_wib = (jam_utc + 7) % 24
radiasi = np.exp(-0.1 * (jam_utc - 12)**2) # Puncak radiasi di UTC jam 12

plt.figure(figsize=(10, 4))
plt.plot(jam_utc, radiasi, label="Radiasi (Skala UTC)", linestyle='--')
plt.plot(jam_wib, radiasi, label="Radiasi (Digeser ke WIB GMT+7)")
plt.title("Hari 10: Pergeseran Puncak Radiasi Matahari (Termodinamika)")
plt.xlabel("Jam (24H)")
plt.legend()
plt.savefig(os.path.join(VISUAL_DIR, "Hari_10_Sinkronisasi_Zona_Waktu.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_10_Sinkronisasi_Zona_Waktu.png"))
