import os, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 2] Simulasi Time-Split Dual Brankas")
sizes = [368880, 68976]
labels = [f'Brankas 1 (Pra-Pelatihan)\n{sizes[0]} Baris', f'Brankas 2 (Penyesuaian)\n{sizes[1]} Baris']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=['#1f77b4', '#ff7f0e'], autopct='%1.1f%%', explode=(0.1, 0))
plt.title("Hari 2: Rasio Time-Split Historis")
plt.savefig(os.path.join(VISUAL_DIR, "Hari_02_Time_Split.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_02_Time_Split.png"))
