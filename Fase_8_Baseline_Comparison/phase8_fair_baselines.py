# =====================================================================
# PHASE 8 – PELATIHAN BASELINE (100% APPLE-TO-APPLE FAIR PLAY EDITION)
# =====================================================================
# Telah disempurnakan agar 100% ADIL:
# 1. Menggunakan data yang SAMA PERSIS (Format 4D, SMOTE Bersih).
# 2. Baseline diberikan akses ke 90 fitur (5 Stasiun x 18 Fitur) yang sama.
# 3. Model baseline akan me-reshape 4D ke 3D secara internal.
# =====================================================================
import os, time, math, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Perangkat: {device}")

# Mount Google Drive
from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except: pass

TENSOR_ROOT = '/content/drive/MyDrive/Riset_ERA5_Land/tensors_mamba'
CLEAN_ROOT  = '/content/drive/MyDrive/Riset_ERA5_Land/clean'
os.makedirs(CLEAN_ROOT, exist_ok=True)

N_FEATURES_TOTAL, HIDDEN_DIM, N_MAMBA_LAYERS, BATCH_SIZE = 90, 192, 4, 64 # 5 stasiun * 18 fitur = 90
EPOCHS_BASE = 300 # [FAIR PLAY] Sama dengan Fase 10 Limit-Breaker
PATIENCE = 50     # [FAIR PLAY] Mencegah baseline hancur karena overfitting

# ---------------------------------------------------------------------
# 1. FUNGSI DATA (MENGGUNAKAN DATA 4D YANG SAMA PERSIS DENGAN FASE 7B)
# ---------------------------------------------------------------------
def get_dataset(prefix):
    # [PERBAIKAN KRUSIAL]: Wajib menggunakan _4d.pt agar adil (SMOTE bersih)
    X = torch.load(f'{TENSOR_ROOT}/{prefix}_X_4d.pt', weights_only=False)
    yr = torch.log1p(torch.load(f'{TENSOR_ROOT}/{prefix}_yr_4d.pt', weights_only=False).clamp(min=0))
    yc = torch.load(f'{TENSOR_ROOT}/{prefix}_yc_4d.pt', weights_only=False)
    return TensorDataset(X, yr, yc)

ds_b1_tr = get_dataset('b1_train'); ds_b2_tr = get_dataset('b2_train')
ds_b1_vl = get_dataset('b1_val'); ds_b2_vl = get_dataset('b2_val')
ds_b1_ts = get_dataset('b1_test'); ds_b2_ts = get_dataset('b2_test')

tr_loader = DataLoader(ConcatDataset([ds_b1_tr, ds_b2_tr]), batch_size=BATCH_SIZE, shuffle=True)
vl_loader = DataLoader(ConcatDataset([ds_b1_vl, ds_b2_vl]), batch_size=BATCH_SIZE, shuffle=False)
ts_loader_bmkg = DataLoader(ds_b2_ts, batch_size=BATCH_SIZE, shuffle=False)
ts_loader_fusi = DataLoader(ConcatDataset([ds_b1_ts, ds_b2_ts]), batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------------------
# 2. BLOK MAMBA (UNTUK ABLASI)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 3. ARSITEKTUR BASELINE KLASIK (FAIR PLAY 90 FITUR)
# ---------------------------------------------------------------------
class CNN_LSTM_Baseline(nn.Module):
    def __init__(self, in_features=90, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.reg_head = nn.Linear(hidden * 2, 1)
        self.cls_head = nn.Linear(hidden * 2, 3)
    def forward(self, x_4d):
        # [FAIR PLAY] Flatten 4D [B, 14, 5, 18] -> 3D [B, 14, 90]
        B, T, G, F_dim = x_4d.shape
        x_3d = x_4d.reshape(B, T, G * F_dim)
        
        x_conv = F.relu(self.conv(x_3d.transpose(1, 2))).transpose(1, 2)
        out, (hn, cn) = self.lstm(x_conv)
        pool = out[:, -1, :]
        return self.reg_head(pool).squeeze(-1), self.cls_head(pool)

class CNN_GRU_Baseline(nn.Module):
    def __init__(self, in_features=90, hidden=64):
        super().__init__()
        self.conv = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.reg_head = nn.Linear(hidden * 2, 1)
        self.cls_head = nn.Linear(hidden * 2, 3)
    def forward(self, x_4d):
        B, T, G, F_dim = x_4d.shape
        x_3d = x_4d.reshape(B, T, G * F_dim)
        
        x_conv = F.relu(self.conv(x_3d.transpose(1, 2))).transpose(1, 2)
        out, hn = self.gru(x_conv)
        pool = out[:, -1, :]
        return self.reg_head(pool).squeeze(-1), self.cls_head(pool)

class ST_Mamba_MLP_Ablation(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(90, HIDDEN_DIM)
        self.layers = nn.ModuleList([MambaBlock(HIDDEN_DIM) for _ in range(N_MAMBA_LAYERS)])
        self.reg_head = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 1))
        self.cls_head = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 3))
    def forward(self, x_4d):
        B, T, G, F_dim = x_4d.shape
        x_3d = x_4d.reshape(B, T, G * F_dim)
        
        x = self.proj(x_3d)
        for layer in self.layers: x = layer(x)
        pool = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=1)
        return self.reg_head(pool).squeeze(-1), self.cls_head(pool)

# ---------------------------------------------------------------------
# 4. FUNGSI PELATIHAN STANDAR (TANPA EVT/PINN/FOCAL DEMI PEMBUKTIAN)
# ---------------------------------------------------------------------
def train_baseline(model, name):
    print(f"\n🚀 Melatih {name} (Maks {EPOCHS_BASE} Epoch, Patience {PATIENCE})...")
    start_t = time.time()
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit_reg = nn.HuberLoss()       # Standar Loss
    crit_cls = nn.CrossEntropyLoss() # Standar Loss

    safe_name = name.split()[0].replace("-", "_")
    save_path = f"{CLEAN_ROOT}/baseline_{safe_name}.pt"

    best_val_loss = float('inf')
    early_stop_counter = 0

    for ep in range(1, EPOCHS_BASE + 1):
        model.train()
        for X, yr, yc in tr_loader:
            X, yr, yc = X.to(device), yr.to(device), yc.to(device)
            opt.zero_grad()
            out_reg, out_cls = model(X)
            loss_r = crit_reg(out_reg, yr)
            loss_c = crit_cls(out_cls, yc)
            loss = loss_r + loss_c
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, yr, yc in vl_loader:
                X, yr, yc = X.to(device), yr.to(device), yc.to(device)
                out_reg, out_cls = model(X)
                loss_r = crit_reg(out_reg, yr)
                loss_c = crit_cls(out_cls, yc)
                val_loss += (loss_r + loss_c).item()

        val_loss /= len(vl_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            if ep % 10 == 0 or ep == 1:
                print(f"   [{name}] Epoch {ep:03d} │ Val Loss: {val_loss:.4f} 💾 (Best)")
        else:
            early_stop_counter += 1
            if ep % 10 == 0:
                print(f"   [{name}] Epoch {ep:03d} │ Val Loss: {val_loss:.4f} (Patience: {early_stop_counter}/{PATIENCE})")

            if early_stop_counter >= PATIENCE:
                print(f"   ⏹️ EARLY STOPPING! {name} berhenti di epoch {ep} karena stagnan.")
                break

    print(f"   📥 Memuat bobot terbaik dari {save_path} untuk evaluasi akhir...")
    model.load_state_dict(torch.load(save_path, weights_only=False))
    model.eval()

    all_reg, all_yt = [], []
    with torch.no_grad():
        for X, yr, _ in ts_loader_bmkg:
            out_reg, _ = model(X.to(device))
            all_reg.append(out_reg.cpu()); all_yt.append(yr)

    vp_mm = torch.expm1(torch.cat(all_reg)).clamp(min=0).numpy().flatten()
    vt_mm = torch.expm1(torch.cat(all_yt)).clamp(min=0).numpy().flatten()
    rmse = np.sqrt(np.mean((vp_mm - vt_mm)**2))

    all_cls, all_yc = [], []
    with torch.no_grad():
        for X, _, yc in ts_loader_fusi:
            _, out_cls = model(X.to(device))
            all_cls.append(out_cls.cpu()); all_yc.append(yc)

    probs = F.softmax(torch.cat(all_cls), dim=1).numpy()
    vt_c = torch.cat(all_yc).numpy()
    vp_c = np.argmax(probs, axis=1)

    acc = accuracy_score(vt_c, vp_c)
    f1 = f1_score(vt_c, vp_c, average='macro')
    cm = confusion_matrix(vt_c, vp_c, labels=[0,1,2])

    print(f"   🏁 HASIL AKHIR {name}:")
    print(f"   ├── Waktu Latih: {(time.time()-start_t):.1f} detik")
    print(f"   ├── RMSE       : {rmse:.2f} mm")
    print(f"   ├── Akurasi    : {acc*100:.2f}%")
    print(f"   └── Macro F1   : {f1:.3f}")

    print("\n   📊 Laporan Klasifikasi Detail:")
    print(classification_report(vt_c, vp_c, target_names=['Aman (0)', 'Waspada (1)', 'Siaga (2)']))
    print("-" * 50)
    return model

# 5. EKSEKUSI
print("\n" + "="*70)
print("🏋️ MEMULAI PELATIHAN BASELINE (100% APPLE-TO-APPLE) UNTUK PERBANDINGAN...")
print("="*70)
model_lstm = train_baseline(CNN_LSTM_Baseline(), "CNN-LSTM (Baseline Klasik 1)")
model_gru  = train_baseline(CNN_GRU_Baseline(), "CNN-GRU (Baseline Klasik 2)")
model_mlp  = train_baseline(ST_Mamba_MLP_Ablation(), "ST-Mamba-MLP (Ablasi GNN & KAN)")
