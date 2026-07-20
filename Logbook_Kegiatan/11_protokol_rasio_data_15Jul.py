import os, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 11] Protokol Rasio Himpunan Data Anti-Kebocoran")
plt.figure(figsize=(6, 6))
plt.pie([70, 15, 15], labels=['Latih (70%)', 'Validasi (15%)', 'Ujian (15%)'], autopct='%1.0f%%', colors=['green', 'gold', 'red'])
plt.title("Hari 11: Distribusi Dataset Bebas Kebocoran Ujian")
plt.savefig(os.path.join(VISUAL_DIR, "Hari_11_Rasio_Data.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_11_Rasio_Data.png"))
