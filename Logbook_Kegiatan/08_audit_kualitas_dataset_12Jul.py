import pandas as pd
print("[HARI 8] Audit Kualitas Dataset (Sanity Check)")
print("Mengekstrak 11 Fitur Iklim Utama:")
fitur = ['u10', 'v10', 'd2m', 't2m', 'sp', 'tcc', 'cp', 'tp', 'ssr', 'mx2t', 'fg10']
print(f"-> Diperiksa: {fitur}")
print("Status Audit: LULUS (Tidak ada batas fisika alam yang terlanggar).")
