# =====================================================================
# PHASE 9 – DASHBOARD EVALUASI FINAL & LEADERBOARD PRESISI (INSTANT)
# =====================================================================
# Script ini secara instan memproses seluruh dataset uji (tanpa loop/animasi)
# dan langsung menyajikan hasil evaluasi final 2-Panel:
# 1) Perbandingan Tren Curah Hujan Test Set (Kiri)
# 2) Leaderboard Akurasi RMSE & MAE Seluruh Model (Kanan)
# Hasil langsung disimpan sebagai berkas cetak PNG publikasi.
# =====================================================================
import os, time, math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# 1. MOUNT GOOGLE DRIVE & KONFIGURASI
from google.colab import drive
try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Perangkat Kerja: {device}")

TENSOR_ROOT = '/content/drive/MyDrive/Riset_ERA5_Land/tensors_mamba'
CLEAN_ROOT  = '/content/drive/MyDrive/Riset_ERA5_Land/clean'
VISUAL_DIR  = '/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi'
os.makedirs(VISUAL_DIR, exist_ok=True)

HIDDEN_DIM, N_MAMBA_LAYERS = 192, 4

# ============================================================
# 2. ARSITEKTUR MODEL
# ============================================================
class CNN_LSTM_Baseline(nn.Module):
    def __init__(self, in_features=90, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.reg_head = nn.Linear(hidden * 2, 1)
        self.cls_head = nn.Linear(hidden * 2, 3)
    def forward(self, x_4d):
        B, T, G, F_dim = x_4d.shape; x_3d = x_4d.reshape(B, T, G * F_dim)
        x_conv = F.relu(self.conv(x_3d.transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(x_conv)
        return self.reg_head(out[:, -1, :]).squeeze(-1), self.cls_head(out[:, -1, :])

class CNN_GRU_Baseline(nn.Module):
    def __init__(self, in_features=90, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.reg_head = nn.Linear(hidden * 2, 1)
        self.cls_head = nn.Linear(hidden * 2, 3)
    def forward(self, x_4d):
        B, T, G, F_dim = x_4d.shape; x_3d = x_4d.reshape(B, T, G * F_dim)
        x_conv = F.relu(self.conv(x_3d.transpose(1, 2))).transpose(1, 2)
        out, _ = self.gru(x_conv)
        return self.reg_head(out[:, -1, :]).squeeze(-1), self.cls_head(out[:, -1, :])

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.2):
        super().__init__()
        self.d_inner = d_model * expand; self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x; x = self.norm(x); xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_path.permute(0, 2, 1))[:, :, :x.shape[1]]
        x_path = F.silu(x_conv.permute(0, 2, 1))
        B, L, D = x_path.shape; A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x_path); delta = F.softplus(self.dt_proj(x_dbl[:, :, :1]))
        B_ssm = x_dbl[:, :, 1:1+self.d_state]; C_ssm = x_dbl[:, :, 1+self.d_state:]
        h = torch.zeros(B, D, self.d_state, device=x.device)
        y = torch.zeros_like(x_path)
        for t in range(L):
            dt = delta[:, t, :]; dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * B_ssm[:, t, :].unsqueeze(1)
            h = dA * h + dB * x_path[:, t, :].unsqueeze(-1)
            y[:, t, :] = (h * C_ssm[:, t, :].unsqueeze(1)).sum(-1) + self.D * x_path[:, t, :]
        return self.dropout(self.out_proj(y * F.silu(z))) + residual

class ST_Mamba_MLP_Ablation(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(90, HIDDEN_DIM)
        self.layers = nn.ModuleList([MambaBlock(HIDDEN_DIM) for _ in range(N_MAMBA_LAYERS)])
        self.reg_head = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 1))
    def forward(self, x_4d):
        B, T, G, F_dim = x_4d.shape; x_3d = x_4d.reshape(B, T, G * F_dim)
        x = self.proj(x_3d)
        for layer in self.layers: x = layer(x)
        pool = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=1)
        return self.reg_head(pool).squeeze(-1)

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=4, spline_order=3):
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        h = 2.0 / grid_size
        self.register_buffer('grid', torch.linspace(-1 - h*spline_order, 1 + h*spline_order, grid_size + 2*spline_order + 1))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.1)
        self.spline_order = spline_order
    def forward(self, x):
        base_out = F.silu(F.linear(x, self.base_weight))
        x_u = torch.tanh(x).unsqueeze(-1)
        bases = ((x_u >= self.grid[:-1]) & (x_u < self.grid[1:])).float()
        for p in range(1, self.spline_order + 1):
            n = bases.shape[-1] - 1
            left = ((x_u - self.grid[:n]) / (self.grid[p:p+n] - self.grid[:n] + 1e-8)) * bases[:, :, :-1]
            right = ((self.grid[p+1:p+1+n] - x_u) / (self.grid[p+1:p+1+n] - self.grid[1:1+n] + 1e-8)) * bases[:, :, 1:]
            bases = left + right
        return base_out + torch.einsum('bin,oin->bo', bases, self.spline_weight)

class Ultimate_GAT_Mamba_KAN(nn.Module):
    def __init__(self, hidden_dim, mamba_layers):
        super().__init__()
        self.proj = nn.Linear(18, hidden_dim)
        self.gat = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.spatial_agg = nn.Linear(5 * hidden_dim, hidden_dim)
        self.mambas = nn.ModuleList([MambaBlock(hidden_dim) for _ in range(mamba_layers)])
        self.head = KANLinear(hidden_dim * 2, 1, grid_size=12)

    def forward(self, x_4d):
        B, T, S, _ = x_4d.shape
        x_proj = self.proj(x_4d)
        x_res = x_proj.reshape(B * T, S, -1)
        x_gat, _ = self.gat(x_res, x_res, x_res)
        x_gat = F.silu(x_gat) + x_res
        x_gat = x_gat.reshape(B, T, -1)
        x_temp = F.relu(self.spatial_agg(x_gat))
        for m in self.mambas: x_temp = m(x_temp)
        x_pool = torch.cat([x_temp.mean(dim=1), x_temp.max(dim=1)[0]], dim=1)
        out = self.head(x_pool)
        return out.squeeze(-1)

# ============================================================
# 3. MEMUAT DATA & MODEL
# ============================================================
print("\n🔄 Memuat Data Uji BMKG...")
X_test = torch.load(f'{TENSOR_ROOT}/b2_test_X_4d.pt', map_location=device, weights_only=False)
yr_test = torch.load(f'{TENSOR_ROOT}/b2_test_yr_4d.pt', map_location=device, weights_only=False).clamp(min=0).cpu().numpy()

# Memuat Model
models = {
    'CNN-LSTM': CNN_LSTM_Baseline().to(device),
    'CNN-GRU': CNN_GRU_Baseline().to(device),
    'ST-Mamba-MLP': ST_Mamba_MLP_Ablation().to(device),
    'ST-Mamba-KAN': Ultimate_GAT_Mamba_KAN(hidden_dim=384, mamba_layers=4).to(device)
}

file_map = {
    'CNN-LSTM': 'baseline_CNN_LSTM.pt', 
    'CNN-GRU': 'baseline_CNN_GRU.pt', 
    'ST-Mamba-MLP': 'baseline_ST_Mamba_MLP.pt'
}

for name, filename in file_map.items():
    p = f'{CLEAN_ROOT}/{filename}'
    if os.path.exists(p):
        models[name].load_state_dict(torch.load(p, map_location=device, weights_only=False), strict=False)
        models[name].eval()
        print(f" ✅ Model {name} siap!")

ckpt_reg_path = f'{CLEAN_ROOT}/ultimate_mamba_kan_reg.pt'
if os.path.exists(ckpt_reg_path):
    ckpt = torch.load(ckpt_reg_path, map_location=device, weights_only=False)
    models['ST-Mamba-KAN'].load_state_dict(ckpt['state'])
    for n, p in models['ST-Mamba-KAN'].named_parameters(): 
        p.data = ckpt['ema'][n]
    models['ST-Mamba-KAN'].eval()
    print(" ✅ Model Utama ST-Mamba-KAN Regressor siap (EMA weights)!")

# ============================================================
# 4. INFERENSI INSTAN (TANPA LOOP DELAY)
# ============================================================
print("\n⚙️ Menjalankan inferensi instan pada seluruh Test Set...")
preds = {name: [] for name in models.keys()}

with torch.no_grad():
    dataset = TensorDataset(X_test)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        for name, model in models.items():
            out = model(x_batch)
            if isinstance(out, tuple):
                out = out[0]
            pred_val = torch.expm1(out).cpu().numpy()
            preds[name].extend(pred_val)

for name in preds.keys():
    preds[name] = np.array(preds[name]).clip(min=0)

# ============================================================
# 5. KALKULASI METRIK EVALUASI AKHIR (TOTAL TEST SET)
# ============================================================
print("\n📊 Menghitung Akurasi Final RMSE & MAE...")
final_metrics = {}
for name in models.keys():
    rmse = np.sqrt(np.mean((yr_test - preds[name])**2))
    mae = np.mean(np.abs(yr_test - preds[name]))
    final_metrics[name] = {'RMSE': rmse, 'MAE': mae}

# Urutkan peringkat
sorted_leaderboard = sorted(final_metrics.items(), key=lambda item: item[1]['RMSE'])
winner_model = sorted_leaderboard[0][0]

# ============================================================
# 6. PENYAJIAN 2-PANEL DASHBOARD FINAL (STATIC & HIGH-QUALITY)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [7, 3]})

# --- PANEL KIRI: Visualisasi Tren Hujan Akhir (100 Hari Terakhir) ---
# Menampilkan 100 hari terakhir dari test set agar grafik bersih dan tidak terlalu padat
plot_days = min(100, len(yr_test))
x_axis = np.arange(len(yr_test) - plot_days, len(yr_test))

axes[0].plot(x_axis, yr_test[-plot_days:], color='black', label='Aktual BMKG (Ground Truth)', linewidth=3.0, marker='o', markersize=4, zorder=5)
axes[0].plot(x_axis, preds['ST-Mamba-KAN'][-plot_days:], color='crimson', label='Prediksi ST-Mamba-KAN (Model Kita)', linewidth=2.5, zorder=6)
axes[0].plot(x_axis, preds['CNN-LSTM'][-plot_days:], color='royalblue', label='Baseline CNN-LSTM', alpha=0.4, linestyle='--')
axes[0].plot(x_axis, preds['CNN-GRU'][-plot_days:], color='darkorange', label='Baseline CNN-GRU', alpha=0.4, linestyle='--')
axes[0].plot(x_axis, preds['ST-Mamba-MLP'][-plot_days:], color='purple', label='Baseline ST-Mamba-MLP', alpha=0.4, linestyle='--')

axes[0].axhline(50, color='red', linestyle=':', alpha=0.8, linewidth=2, label='Batas Bahaya Siaga (50 mm)')

axes[0].set_title(f"Tren Curah Hujan Aktual vs Prediksi Model (Sampel 100 Hari Terakhir)", fontsize=13, fontweight='bold', color='black')
axes[0].set_xlabel("Hari Pengamatan (Test Set)", fontsize=11)
axes[0].set_ylabel("Curah Hujan (mm / Hari)", fontsize=11)
axes[0].set_ylim(-5, max(yr_test[-plot_days:].max(), preds['ST-Mamba-KAN'][-plot_days:].max()) + 15)
axes[0].legend(loc='upper left', frameon=True, shadow=True, facecolor='white')
axes[0].grid(True, alpha=0.15)

# --- PANEL KANAN: Leaderboard Akurasi Seluruh Model ---
names_sorted = [item[0] for item in sorted_leaderboard]
rmse_values = [item[1]['RMSE'] for item in sorted_leaderboard]
mae_values = [item[1]['MAE'] for item in sorted_leaderboard]
colors = ['gold' if n == winner_model else '#9fbcdb' for n in names_sorted]

bars = axes[1].barh(names_sorted, rmse_values, color=colors, edgecolor='black', height=0.6)
axes[1].invert_yaxis()

for idx, bar in enumerate(bars):
    rmse_val = rmse_values[idx]
    mae_val = mae_values[idx]
    axes[1].text(rmse_val + 0.2, bar.get_y() + bar.get_height()/2, 
                 f'RMSE: {rmse_val:.2f} | MAE: {mae_val:.2f} mm', 
                 va='center', ha='left', fontsize=10.5, fontweight='bold')
                 
axes[1].set_title("🏆 LEADERBOARD EVALUASI FINAL\n(RMSE & MAE Terkecil = Terbaik)", fontsize=13, fontweight='bold', color='navy')
axes[1].set_xlabel("RMSE Error (Makin Kecil Makin Presisi)", fontsize=11)
axes[1].set_xlim(0, max(rmse_values) + 12)
axes[1].grid(True, alpha=0.15, axis='x')

# Tambahkan label pemenang dengan Mahkota
axes[1].text(0.5, -0.6, f"👑 WINNER MODEL:\n{winner_model}", 
             fontsize=13, color='darkgoldenrod', weight='bold', 
             ha='center', va='center', transform=axes[1].transData,
             bbox=dict(facecolor='#fffde6', edgecolor='gold', boxstyle='round,pad=0.6'))

plt.tight_layout()

# Simpan sebagai file PNG berkualitas tinggi
print("\n💾 Menyimpan berkas visualisasi final ke Google Drive...")
final_img_path = os.path.join(VISUAL_DIR, "Hari_10_Final_Evaluation_Leaderboard.png")
plt.savefig(final_img_path, dpi=300)
plt.show()

# ============================================================
# 7. CETAK TABEL METRIK FINAL DI KONSOL
# ============================================================
print("\n" + "="*85)
print(f"🏆 HASIL EVALUASI AKHIR & PERINGKAT MODEL (DEFINITIF)")
print("="*85)
for rank, (name, metrics) in enumerate(sorted_leaderboard, 1):
    medal = "🥇 [WINNER]" if rank == 1 else f"🥈 [Rank {rank}]" if rank == 2 else f"🥉 [Rank {rank}]"
    print(f" {medal:<12} | Model: {name:<13} | RMSE: {metrics['RMSE']:.3f} mm | MAE: {metrics['MAE']:.3f} mm")
print("="*85)
print(f"-> Gambar cetak resolusi tinggi disimpan di: {final_img_path}")
print("="*85)