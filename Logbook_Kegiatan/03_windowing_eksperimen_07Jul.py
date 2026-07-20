import os, matplotlib.pyplot as plt

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Evaluasi Windowing 72 Jam vs 336 Jam")
x = ['72 Jam', '336 Jam']
memory = [2.4, 11.2] 
akurasi = [75.4, 88.0] 

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.bar(x, memory, color='red', alpha=0.6, width=0.4, label='Konsumsi RAM (GB)')
ax2.plot(x, akurasi, color='blue', marker='o', linewidth=2, label='Akurasi Deteksi (%)')

ax1.set_ylabel('RAM (GB)', color='red')
ax2.set_ylabel('Akurasi (%)', color='blue')
plt.title("Trade-off Memori vs Akurasi (Windowing)")
output_path = os.path.join(VISUAL_DIR, "Hari_03_Windowing_Eksperimen.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
