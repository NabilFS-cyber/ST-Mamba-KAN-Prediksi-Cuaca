import os, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 3] Evaluasi Windowing 72 Jam vs 336 Jam")
x = ['72 Jam (3 Hari)', '336 Jam (14 Hari)']
memory = [2.4, 11.2] # dalam GB (Simulasi)
akurasi = [75.4, 88.0] # dalam persen (Simulasi)

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.bar(x, memory, color='red', alpha=0.6, width=0.4, label='Konsumsi RAM (GB)')
ax2.plot(x, akurasi, color='blue', marker='o', linewidth=2, label='Akurasi Deteksi (%)')

ax1.set_ylabel('RAM (GB)', color='red')
ax2.set_ylabel('Akurasi (%)', color='blue')
plt.title("Hari 3: Trade-off Memori vs Akurasi (Windowing)")
plt.savefig(os.path.join(VISUAL_DIR, "Hari_03_Windowing_Eksperimen.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_03_Windowing_Eksperimen.png"))
