"""
train_local.py — Entraînement local GRU (Approche A ou B)
══════════════════════════════════════════════════════
Fixes v2 :
  - API torch.amp (non dépréciée) pour GradScaler et autocast
  - Gradient clipping (max_norm=1.0) → évite les explosions → nan
  - pos_weight en FP32 forcé → évite overflow FP16
  - Détection NaN dans les données au démarrage
  - Checkpoint protégé : sauvegarde dès epoch 1 garantie
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from models.fraud_rnn import FraudRNN
from data.dataloader import get_dataloaders_approach_A, get_dataloaders_approach_B

# ══════════════════════════════════════════════════════
# ⚙️  CONFIGURATION
# ══════════════════════════════════════════════════════

DATA_DIR       = "data/node_1/tensors/"
CHECKPOINT_DIR = "checkpoints/"

# Modèle
INPUT_DIM    = 26
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT_RATE = 0.2

# Entraînement
BATCH_SIZE    = 256
SEQ_LEN       = 5
TRAIN_RATIO   = 0.8
POS_WEIGHT    = 167.0   # ratio réel : 362 974 légit / 2 174 fraudes
PATIENCE      = 5       # early stopping
APPROACH      = "B"     # "A" = intra-user temporel | "B" = population unseen users

EPOCHS        = 20
LEARNING_RATE = 0.0005

# ══════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # Mixed Precision uniquement sur GPU

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME, GPU_MEM = "N/A", 0.0


# ══════════════════════════════════════════════════════
# 🔍 VÉRIFICATION DES TENSEURS
# ══════════════════════════════════════════════════════

def check_tensors(data_dir: str) -> bool:
    """Vérifie l'absence de NaN/Inf dans les tenseurs avant l'entraînement."""
    files = sorted(Path(data_dir).glob("X_batch_*.npy"))
    total_nan = total_inf = 0

    for f in files:
        X = np.load(f)
        total_nan += np.isnan(X).sum()
        total_inf += np.isinf(X).sum()

    if total_nan > 0 or total_inf > 0:
        print(f"❌ Tenseurs corrompus → NaN:{total_nan}  Inf:{total_inf}")
        print("   Relance generate_tensors_from_csv.py pour régénérer les tenseurs.")
        return False

    print(f"✅ Tenseurs vérifiés ({len(files)} fichiers) — aucun NaN/Inf")
    return True


# ══════════════════════════════════════════════════════
# 🔧 UTILS
# ══════════════════════════════════════════════════════

def save_checkpoint(model, epoch, metric_val, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "metric"     : metric_val,
    }, path)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # AMP désactivé en évaluation :
            # pos_weight=167 en FP16 → overflow → NaN dans la loss
            logits = model(x).float()   # forcé FP32
            loss   = criterion(logits, y.float())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss  += loss.item() * y.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    n = len(all_targets)
    return (
        total_loss / n if n > 0 else 0.0,
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


# ══════════════════════════════════════════════════════
# 🚀 TRAIN
# ══════════════════════════════════════════════════════

def train_and_evaluate():
    print(f"\n{'═'*55}")
    approach_label = "A — Split Temporel Intra-User 70/10/20" if APPROACH == "A" else "B — Population Unseen Users 80/20"
    print(f"  🚀  Entraînement GRU — Approche {approach_label}")
    print(f"{'═'*55}")
    print(f"  Device  : {DEVICE}  ({GPU_NAME}  {GPU_MEM:.1f} GB)" if DEVICE.type == "cuda"
          else f"  Device  : CPU")
    print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
    print(f"  GRU     : hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  dropout={DROPOUT_RATE}")
    print(f"  pos_weight : {POS_WEIGHT}  |  Mixed Precision : {USE_AMP}")
    print(f"{'─'*55}\n")

    # ── Vérification des tenseurs ─────────────────────────────────
    if not check_tensors(DATA_DIR):
        return

    # ── 1. Données ────────────────────────────────────────────────
    if APPROACH == "A":
        print("\n📂 Chargement (Approche A — Split Temporel Intra-User 70/10/20)...")
        train_loader, val_loader, test_loader = get_dataloaders_approach_A(
            data_dir    = DATA_DIR,
            train_ratio = 0.70,
            val_ratio   = 0.10,
            batch_size  = BATCH_SIZE,
            seq_len     = SEQ_LEN,
        )
    else:
        print("\n📂 Chargement (Approche B — Population Unseen Users 80/20)...")
        train_loader, val_loader, test_loader = get_dataloaders_approach_B(
            data_dir    = DATA_DIR,
            train_ratio = 0.70,    # 70% train
            val_ratio   = 0.10,    # 10% validation (early stopping)
            batch_size  = BATCH_SIZE,
            seq_len     = SEQ_LEN,
        )

    # ── 2. Modèle ─────────────────────────────────────────────────
    model = FraudRNN(
        input_dim    = INPUT_DIM,
        hidden_dim   = HIDDEN_DIM,
        num_layers   = NUM_LAYERS,
        dropout_rate = DROPOUT_RATE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Paramètres entraînables : {total_params:,}")

    # ── 3. Loss / Optimizer / Scheduler ──────────────────────────
    # FIX : pos_weight en float() explicite → reste FP32 même avec AMP
    pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ✅ API non dépréciée : torch.amp.GradScaler
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

    # ── 4. Boucle d'entraînement ──────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    checkpoint_path   = f"{CHECKPOINT_DIR}best_gru_approach{APPROACH}.pt"

    print(f"\n{'─'*55}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        n_train          = 0
        has_nan          = False

        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward en FP16 (vitesse), loss en FP32 (stabilité)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(x)
            logits_fp32 = logits.float()   # repasse en FP32 avant la loss
            loss = criterion(logits_fp32, y.float())

            # Détection NaN dans la loss
            if torch.isnan(loss):
                has_nan = True
                continue

            loss.backward()   # pas de scaler car loss est en FP32

            # Gradient clipping → évite les explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item() * y.size(0)
            n_train          += y.size(0)

        if has_nan:
            print(f"⚠️  Epoch {epoch} : NaN détectés dans la loss → batches ignorés")

        avg_train_loss = total_train_loss / n_train if n_train > 0 else float("nan")
        val_loss, _, _, _ = evaluate(model, val_loader,  criterion, DEVICE)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch:>2}/{EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| LR: {current_lr:.6f}")

        # Checkpoint — sauvegarde si meilleure val_loss ET loss valide
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, epoch, val_loss, checkpoint_path)
            print(f"  💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Pas d'amélioration ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping à l'epoch {epoch}.")
            break

    # ── 5. Évaluation finale ──────────────────────────────────────
    print(f"\n{'═'*55}")
    print("📊 Évaluation finale — meilleur modèle")
    print(f"{'═'*55}")

    if not Path(checkpoint_path).exists():
        print("❌ Aucun checkpoint sauvegardé (toutes les losses étaient NaN).")
        print("   → Vérifie tes tenseurs avec check_tensors.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    print(f"   Chargé depuis epoch {checkpoint['epoch']}  "
          f"(val_loss={checkpoint['metric']:.4f})\n")

    from sklearn.metrics import f1_score

    # ── Calibration du seuil sur VALIDATION (pas sur test) ───────────────────
    # Approche A : val_loader = 10% du milieu (clients connus, période intermédiaire)
    # Approche B : val_loader = 10% des users (clients inconnus, jamais vus)
    # Dans les deux cas : le seuil est cherché sur val → appliqué au test
    # → aucun biais introduit sur les métriques finales du test
    print("🎯 Calibration du seuil sur la validation...")
    _, val_targets_cal, _, val_probs_cal = evaluate(model, val_loader, criterion, DEVICE)

    best_f1_val, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds_t = (val_probs_cal >= t).astype(float)
        f1 = f1_score(val_targets_cal, preds_t, pos_label=1, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val, best_thresh = f1, t
    print(f"   Seuil optimal trouvé : {best_thresh:.2f}  (F1 val={best_f1_val:.4f})")
    print(f"   Ce seuil sera appliqué au test (non biaisé)\n")

    # ── Évaluation finale sur TEST ────────────────────────────────────────────
    _, all_targets, all_preds, all_probs = evaluate(model, test_loader, criterion, DEVICE)

    # Rapport avec seuil par défaut 0.5
    print("📋 RAPPORT DE CLASSIFICATION (seuil 0.5)")
    print("─"*55)
    print(classification_report(
        all_targets, all_preds,
        target_names=["Légitime (0)", "Fraude (1)"],
        zero_division=0,
    ))

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print("🧩 MATRICE DE CONFUSION (seuil 0.5)")
    print("─"*55)
    print(f"  Vrais Négatifs (Légitimes détectés) : {tn:>7,}")
    print(f"  Faux Positifs  (Fausses alertes)    : {fp:>7,} ⚠️")
    print(f"  Faux Négatifs  (Fraudes manquées)   : {fn:>7,} 🚨")
    print(f"  Vrais Positifs (Fraudes détectées)  : {tp:>7,}")

    print("\n📈 MÉTRIQUES DE PERFORMANCE")
    print("─"*55)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
        pr_auc  = average_precision_score(all_targets, all_probs)
        print(f"  🌟 AUC-ROC                   : {roc_auc:.4f}")
        print(f"  🎯 PR-AUC (Precision-Recall) : {pr_auc:.4f}")

        # ── Seuil calibré sur val → appliqué au test ─────────────────────────
        preds_cal = (all_probs >= best_thresh).astype(float)
        cm_cal    = confusion_matrix(all_targets, preds_cal)
        tn_c, fp_c, fn_c, tp_c = cm_cal.ravel()
        f1_cal = f1_score(all_targets, preds_cal, pos_label=1, zero_division=0)

        print(f"\n  🔍 SEUIL CALIBRÉ SUR VALIDATION → appliqué au test")
        print(f"  Seuil  : {best_thresh:.2f}  (calibré sur val, F1 val={best_f1_val:.4f})")
        print(f"  F1-fraude sur test : {f1_cal:.4f}")
        print(f"\n  Avec seuil calibré :")
        print(f"  Vrais Positifs (Fraudes détectées) : {tp_c:>6,}  / {int(all_targets.sum())}")
        print(f"  Faux Positifs  (Fausses alertes)   : {fp_c:>6,}  ← à minimiser")
        print(f"  Faux Négatifs  (Fraudes manquées)  : {fn_c:>6,}")
        print(f"\n  Rappel fraude   : {tp_c / max(tp_c+fn_c,1):.2%}")
        print(f"  Précision fraude: {tp_c / max(tp_c+fp_c,1):.2%}")

        # ── Comparaison seuil 0.5 vs seuil calibré ───────────────────────────
        print(f"\n  📊 Comparaison seuils :")
        print(f"  {'':20s} {'Seuil 0.5':>12} {'Seuil calibré':>14}")
        print(f"  {'Fausses alertes':20s} {fp:>12,} {fp_c:>14,}")
        print(f"  {'Fraudes détectées':20s} {tp:>12,} {tp_c:>14,}")
        print(f"  {'Rappel fraude':20s} {tp/max(tp+fn,1):>12.2%} {tp_c/max(tp_c+fn_c,1):>14.2%}")
        print(f"  {'Précision fraude':20s} {tp/max(tp+fp,1):>12.2%} {tp_c/max(tp_c+fp_c,1):>14.2%}")

    except ValueError as e:
        print(f"  ⚠️ Calcul AUC impossible : {e}")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    train_and_evaluate()





# """
# train_local.py — Entraînement local GRU (Approche A ou B)
# ══════════════════════════════════════════════════════
# Fixes v2 :
#   - API torch.amp (non dépréciée) pour GradScaler et autocast
#   - Gradient clipping (max_norm=1.0) → évite les explosions → nan
#   - pos_weight en FP32 forcé → évite overflow FP16
#   - Détection NaN dans les données au démarrage
#   - Checkpoint protégé : sauvegarde dès epoch 1 garantie
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from pathlib import Path
# from sklearn.metrics import (
#     classification_report,
#     roc_auc_score,
#     confusion_matrix,
#     average_precision_score,
# )

# from models.fraud_rnn import FraudRNN
# from data.dataloader import get_dataloaders_approach_A, get_dataloaders_approach_B

# # ══════════════════════════════════════════════════════
# # ⚙️  CONFIGURATION
# # ══════════════════════════════════════════════════════

# DATA_DIR       = "data/node_1/tensors/"
# CHECKPOINT_DIR = "checkpoints/"

# # Modèle
# INPUT_DIM    = 26
# HIDDEN_DIM   = 128
# NUM_LAYERS   = 2
# DROPOUT_RATE = 0.2

# # Entraînement
# BATCH_SIZE    = 256
# SEQ_LEN       = 5
# TRAIN_RATIO   = 0.8
# POS_WEIGHT    = 167.0   # ratio réel : 362 974 légit / 2 174 fraudes
# PATIENCE      = 5       # early stopping
# APPROACH      = "B"     # "A" = intra-user temporel | "B" = population unseen users

# EPOCHS        = 20
# LEARNING_RATE = 0.0005

# # ══════════════════════════════════════════════════════

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# USE_AMP = DEVICE.type == "cuda"   # Mixed Precision uniquement sur GPU

# if DEVICE.type == "cuda":
#     torch.backends.cudnn.benchmark = True
#     GPU_NAME = torch.cuda.get_device_name(0)
#     GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
# else:
#     GPU_NAME, GPU_MEM = "N/A", 0.0


# # ══════════════════════════════════════════════════════
# # 🔍 VÉRIFICATION DES TENSEURS
# # ══════════════════════════════════════════════════════

# def check_tensors(data_dir: str) -> bool:
#     """Vérifie l'absence de NaN/Inf dans les tenseurs avant l'entraînement."""
#     files = sorted(Path(data_dir).glob("X_batch_*.npy"))
#     total_nan = total_inf = 0

#     for f in files:
#         X = np.load(f)
#         total_nan += np.isnan(X).sum()
#         total_inf += np.isinf(X).sum()

#     if total_nan > 0 or total_inf > 0:
#         print(f"❌ Tenseurs corrompus → NaN:{total_nan}  Inf:{total_inf}")
#         print("   Relance generate_tensors_from_csv.py pour régénérer les tenseurs.")
#         return False

#     print(f"✅ Tenseurs vérifiés ({len(files)} fichiers) — aucun NaN/Inf")
#     return True


# # ══════════════════════════════════════════════════════
# # 🔧 UTILS
# # ══════════════════════════════════════════════════════

# def save_checkpoint(model, epoch, metric_val, path):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "epoch"      : epoch,
#         "model_state": model.state_dict(),
#         "metric"     : metric_val,
#     }, path)

# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss  = 0.0
#     all_preds, all_targets, all_probs = [], [], []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             # AMP désactivé en évaluation :
#             # pos_weight=167 en FP16 → overflow → NaN dans la loss
#             logits = model(x).float()   # forcé FP32
#             loss   = criterion(logits, y.float())

#             probs = torch.sigmoid(logits)
#             preds = (probs > 0.5).float()

#             total_loss  += loss.item() * y.size(0)
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(y.cpu().numpy())

#     n = len(all_targets)
#     return (
#         total_loss / n if n > 0 else 0.0,
#         np.array(all_targets),
#         np.array(all_preds),
#         np.array(all_probs),
#     )


# # ══════════════════════════════════════════════════════
# # 🚀 TRAIN
# # ══════════════════════════════════════════════════════

# def train_and_evaluate():
#     print(f"\n{'═'*55}")
#     approach_label = "A — Split Temporel Intra-User 70/10/20" if APPROACH == "A" else "B — Population Unseen Users 80/20"
#     print(f"  🚀  Entraînement GRU — Approche {approach_label}")
#     print(f"{'═'*55}")
#     print(f"  Device  : {DEVICE}  ({GPU_NAME}  {GPU_MEM:.1f} GB)" if DEVICE.type == "cuda"
#           else f"  Device  : CPU")
#     print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
#     print(f"  GRU     : hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  dropout={DROPOUT_RATE}")
#     print(f"  pos_weight : {POS_WEIGHT}  |  Mixed Precision : {USE_AMP}")
#     print(f"{'─'*55}\n")

#     # ── Vérification des tenseurs ─────────────────────────────────
#     if not check_tensors(DATA_DIR):
#         return

#     # ── 1. Données ────────────────────────────────────────────────
#     if APPROACH == "A":
#         print("\n📂 Chargement (Approche A — Split Temporel Intra-User 70/10/20)...")
#         train_loader, val_loader, test_loader = get_dataloaders_approach_A(
#             data_dir    = DATA_DIR,
#             train_ratio = 0.70,
#             val_ratio   = 0.10,
#             batch_size  = BATCH_SIZE,
#             seq_len     = SEQ_LEN,
#         )
#     else:
#         print("\n📂 Chargement (Approche B — Population Unseen Users 80/20)...")
#         train_loader, test_loader = get_dataloaders_approach_B(
#             data_dir    = DATA_DIR,
#             train_ratio = 0.80,
#             batch_size  = BATCH_SIZE,
#             seq_len     = SEQ_LEN,
#         )
#         val_loader = test_loader   # early stopping sur test (pas de val séparé en B)

#     # ── 2. Modèle ─────────────────────────────────────────────────
#     model = FraudRNN(
#         input_dim    = INPUT_DIM,
#         hidden_dim   = HIDDEN_DIM,
#         num_layers   = NUM_LAYERS,
#         dropout_rate = DROPOUT_RATE,
#     ).to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n🧠 Paramètres entraînables : {total_params:,}")

#     # ── 3. Loss / Optimizer / Scheduler ──────────────────────────
#     # FIX : pos_weight en float() explicite → reste FP32 même avec AMP
#     pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
#     criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=2
#     )

#     # ✅ API non dépréciée : torch.amp.GradScaler
#     scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

#     # ── 4. Boucle d'entraînement ──────────────────────────────────
#     best_val_loss     = float("inf")
#     epochs_no_improve = 0
#     checkpoint_path   = f"{CHECKPOINT_DIR}best_gru_approach{APPROACH}.pt"

#     print(f"\n{'─'*55}")

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         total_train_loss = 0.0
#         n_train          = 0
#         has_nan          = False

#         for x, y in train_loader:
#             x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             # Forward en FP16 (vitesse), loss en FP32 (stabilité)
#             with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
#                 logits = model(x)
#             logits_fp32 = logits.float()   # repasse en FP32 avant la loss
#             loss = criterion(logits_fp32, y.float())

#             # Détection NaN dans la loss
#             if torch.isnan(loss):
#                 has_nan = True
#                 continue

#             loss.backward()   # pas de scaler car loss est en FP32

#             # Gradient clipping → évite les explosions
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()

#             total_train_loss += loss.item() * y.size(0)
#             n_train          += y.size(0)

#         if has_nan:
#             print(f"⚠️  Epoch {epoch} : NaN détectés dans la loss → batches ignorés")

#         avg_train_loss = total_train_loss / n_train if n_train > 0 else float("nan")
#         val_loss, _, _, _ = evaluate(model, val_loader,  criterion, DEVICE)

#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]["lr"]

#         print(f"Epoch [{epoch:>2}/{EPOCHS}] "
#               f"| Train Loss: {avg_train_loss:.4f} "
#               f"| Val Loss: {val_loss:.4f} "
#               f"| LR: {current_lr:.6f}")

#         # Checkpoint — sauvegarde si meilleure val_loss ET loss valide
#         if not np.isnan(val_loss) and val_loss < best_val_loss:
#             best_val_loss     = val_loss
#             epochs_no_improve = 0
#             save_checkpoint(model, epoch, val_loss, checkpoint_path)
#             print(f"  💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
#         else:
#             epochs_no_improve += 1
#             print(f"  ⏳ Pas d'amélioration ({epochs_no_improve}/{PATIENCE})")

#         if epochs_no_improve >= PATIENCE:
#             print(f"\n🛑 Early stopping à l'epoch {epoch}.")
#             break

#     # ── 5. Évaluation finale ──────────────────────────────────────
#     print(f"\n{'═'*55}")
#     print("📊 Évaluation finale — meilleur modèle")
#     print(f"{'═'*55}")

#     if not Path(checkpoint_path).exists():
#         print("❌ Aucun checkpoint sauvegardé (toutes les losses étaient NaN).")
#         print("   → Vérifie tes tenseurs avec check_tensors.py")
#         return

#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
#     model.load_state_dict(checkpoint["model_state"])
#     print(f"   Chargé depuis epoch {checkpoint['epoch']}  "
#           f"(val_loss={checkpoint['metric']:.4f})\n")

#     _, all_targets, all_preds, all_probs = evaluate(model, test_loader, criterion, DEVICE)

#     print("📋 RAPPORT DE CLASSIFICATION")
#     print("─"*55)
#     print(classification_report(
#         all_targets, all_preds,
#         target_names=["Légitime (0)", "Fraude (1)"],
#         zero_division=0,
#     ))

#     cm = confusion_matrix(all_targets, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     print("🧩 MATRICE DE CONFUSION")
#     print("─"*55)
#     print(f"  Vrais Négatifs (Légitimes détectés) : {tn:>7,}")
#     print(f"  Faux Positifs  (Fausses alertes)    : {fp:>7,} ⚠️")
#     print(f"  Faux Négatifs  (Fraudes manquées)   : {fn:>7,} 🚨")
#     print(f"  Vrais Positifs (Fraudes détectées)  : {tp:>7,}")

#     print("\n📈 MÉTRIQUES DE PERFORMANCE")
#     print("─"*55)
#     try:
#         from sklearn.metrics import roc_curve, f1_score
#         roc_auc = roc_auc_score(all_targets, all_probs)
#         pr_auc  = average_precision_score(all_targets, all_probs)
#         print(f"  🌟 AUC-ROC                   : {roc_auc:.4f}")
#         print(f"  🎯 PR-AUC (Precision-Recall) : {pr_auc:.4f}")

#         # Recherche du seuil optimal (max F1 sur la fraude)
#         thresholds_to_try = np.arange(0.05, 0.95, 0.01)
#         best_f1, best_thresh = 0.0, 0.5
#         for t in thresholds_to_try:
#             preds_t = (all_probs >= t).astype(float)
#             f1 = f1_score(all_targets, preds_t, pos_label=1, zero_division=0)
#             if f1 > best_f1:
#                 best_f1, best_thresh = f1, t

#         print(f"\n  🔍 SEUIL OPTIMAL (max F1-fraude)")
#         print(f"  Seuil  : {best_thresh:.2f}  (au lieu de 0.50)")
#         print(f"  F1-fraude : {best_f1:.4f}")

#         preds_opt = (all_probs >= best_thresh).astype(float)
#         cm_opt = confusion_matrix(all_targets, preds_opt)
#         tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
#         print(f"\n  Avec seuil optimal :")
#         print(f"  Vrais Positifs (Fraudes détectées) : {tp_o:>6,}  / {int(all_targets.sum())}")
#         print(f"  Faux Positifs  (Fausses alertes)   : {fp_o:>6,}")
#         print(f"  Faux Négatifs  (Fraudes manquées)  : {fn_o:>6,}")
#         print(f"\n  Rappel fraude  : {tp_o / max(tp_o+fn_o,1):.2%}")
#         print(f"  Précision fraude: {tp_o / max(tp_o+fp_o,1):.2%}")

#     except ValueError as e:
#         print(f"  ⚠️ Calcul AUC impossible : {e}")

#     print(f"\n{'═'*55}\n")


# if __name__ == "__main__":
#     train_and_evaluate()







# """
# train_local.py — Entraînement local GRU + Approche A
# ══════════════════════════════════════════════════════
# Fixes v2 :
#   - API torch.amp (non dépréciée) pour GradScaler et autocast
#   - Gradient clipping (max_norm=1.0) → évite les explosions → nan
#   - pos_weight en FP32 forcé → évite overflow FP16
#   - Détection NaN dans les données au démarrage
#   - Checkpoint protégé : sauvegarde dès epoch 1 garantie
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from pathlib import Path
# from sklearn.metrics import (
#     classification_report,
#     roc_auc_score,
#     confusion_matrix,
#     average_precision_score,
# )

# from models.fraud_rnn import FraudRNN
# from data.dataloader import get_dataloaders_approach_A, get_dataloaders_approach_B

# # ══════════════════════════════════════════════════════
# # ⚙️  CONFIGURATION
# # ══════════════════════════════════════════════════════

# DATA_DIR       = "data/node_1/tensors/"
# CHECKPOINT_DIR = "checkpoints/"

# # Modèle
# INPUT_DIM    = 26
# HIDDEN_DIM   = 128
# NUM_LAYERS   = 2
# DROPOUT_RATE = 0.2

# # Entraînement
# EPOCHS        = 20
# BATCH_SIZE    = 256
# LEARNING_RATE = 0.0005
# SEQ_LEN       = 5
# TRAIN_RATIO   = 0.8
# POS_WEIGHT    = 167.0   # ratio réel : 362 974 légit / 2 174 fraudes
# PATIENCE      = 5       # early stopping

# # ══════════════════════════════════════════════════════

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# USE_AMP = DEVICE.type == "cuda"   # Mixed Precision uniquement sur GPU

# if DEVICE.type == "cuda":
#     torch.backends.cudnn.benchmark = True
#     GPU_NAME = torch.cuda.get_device_name(0)
#     GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
# else:
#     GPU_NAME, GPU_MEM = "N/A", 0.0


# # ══════════════════════════════════════════════════════
# # 🔍 VÉRIFICATION DES TENSEURS
# # ══════════════════════════════════════════════════════

# def check_tensors(data_dir: str) -> bool:
#     """Vérifie l'absence de NaN/Inf dans les tenseurs avant l'entraînement."""
#     files = sorted(Path(data_dir).glob("X_batch_*.npy"))
#     total_nan = total_inf = 0

#     for f in files:
#         X = np.load(f)
#         total_nan += np.isnan(X).sum()
#         total_inf += np.isinf(X).sum()

#     if total_nan > 0 or total_inf > 0:
#         print(f"❌ Tenseurs corrompus → NaN:{total_nan}  Inf:{total_inf}")
#         print("   Relance generate_tensors_from_csv.py pour régénérer les tenseurs.")
#         return False

#     print(f"✅ Tenseurs vérifiés ({len(files)} fichiers) — aucun NaN/Inf")
#     return True


# # ══════════════════════════════════════════════════════
# # 🔧 UTILS
# # ══════════════════════════════════════════════════════

# def save_checkpoint(model, epoch, metric_val, path):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "epoch"      : epoch,
#         "model_state": model.state_dict(),
#         "metric"     : metric_val,
#     }, path)

# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss  = 0.0
#     all_preds, all_targets, all_probs = [], [], []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             # AMP désactivé en évaluation :
#             # pos_weight=167 en FP16 → overflow → NaN dans la loss
#             logits = model(x).float()   # forcé FP32
#             loss   = criterion(logits, y.float())

#             probs = torch.sigmoid(logits)
#             preds = (probs > 0.5).float()

#             total_loss  += loss.item() * y.size(0)
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(y.cpu().numpy())

#     n = len(all_targets)
#     return (
#         total_loss / n if n > 0 else 0.0,
#         np.array(all_targets),
#         np.array(all_preds),
#         np.array(all_probs),
#     )


# # ══════════════════════════════════════════════════════
# # 🚀 TRAIN
# # ══════════════════════════════════════════════════════

# def train_and_evaluate():
#     print(f"\n{'═'*55}")
#     print(f"  🚀  Entraînement GRU — Approche A")
#     print(f"{'═'*55}")
#     print(f"  Device  : {DEVICE}  ({GPU_NAME}  {GPU_MEM:.1f} GB)" if DEVICE.type == "cuda"
#           else f"  Device  : CPU")
#     print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
#     print(f"  GRU     : hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  dropout={DROPOUT_RATE}")
#     print(f"  pos_weight : {POS_WEIGHT}  |  Mixed Precision : {USE_AMP}")
#     print(f"{'─'*55}\n")

#     # ── Vérification des tenseurs ─────────────────────────────────
#     if not check_tensors(DATA_DIR):
#         return

#     # ── 1. Données ────────────────────────────────────────────────
#     print("\n📂 Chargement (Approche A — Split Temporel Intra-User)...")
#     # Approche A : 3 loaders — train(70%) / val(10%) / test(20%)
#     train_loader, val_loader, test_loader = get_dataloaders_approach_A(
#         data_dir    = DATA_DIR,
#         train_ratio = 0.70,     # 70% train
#         val_ratio   = 0.10,     # 10% validation (early stopping)
#         batch_size  = BATCH_SIZE,
#         seq_len     = SEQ_LEN,
#     )

#     # ── 2. Modèle ─────────────────────────────────────────────────
#     model = FraudRNN(
#         input_dim    = INPUT_DIM,
#         hidden_dim   = HIDDEN_DIM,
#         num_layers   = NUM_LAYERS,
#         dropout_rate = DROPOUT_RATE,
#     ).to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n🧠 Paramètres entraînables : {total_params:,}")

#     # ── 3. Loss / Optimizer / Scheduler ──────────────────────────
#     # FIX : pos_weight en float() explicite → reste FP32 même avec AMP
#     pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
#     criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=2
#     )

#     # ✅ API non dépréciée : torch.amp.GradScaler
#     scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

#     # ── 4. Boucle d'entraînement ──────────────────────────────────
#     best_val_loss     = float("inf")
#     epochs_no_improve = 0
#     checkpoint_path   = f"{CHECKPOINT_DIR}best_gru_approachA.pt"

#     print(f"\n{'─'*55}")

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         total_train_loss = 0.0
#         n_train          = 0
#         has_nan          = False

#         for x, y in train_loader:
#             x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             # Forward en FP16 (vitesse), loss en FP32 (stabilité)
#             with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
#                 logits = model(x)
#             logits_fp32 = logits.float()   # repasse en FP32 avant la loss
#             loss = criterion(logits_fp32, y.float())

#             # Détection NaN dans la loss
#             if torch.isnan(loss):
#                 has_nan = True
#                 continue

#             loss.backward()   # pas de scaler car loss est en FP32

#             # Gradient clipping → évite les explosions
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()

#             total_train_loss += loss.item() * y.size(0)
#             n_train          += y.size(0)

#         if has_nan:
#             print(f"⚠️  Epoch {epoch} : NaN détectés dans la loss → batches ignorés")

#         avg_train_loss = total_train_loss / n_train if n_train > 0 else float("nan")
#         val_loss, _, _, _ = evaluate(model, val_loader,  criterion, DEVICE)

#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]["lr"]

#         print(f"Epoch [{epoch:>2}/{EPOCHS}] "
#               f"| Train Loss: {avg_train_loss:.4f} "
#               f"| Val Loss: {val_loss:.4f} "
#               f"| LR: {current_lr:.6f}")

#         # Checkpoint — sauvegarde si meilleure val_loss ET loss valide
#         if not np.isnan(val_loss) and val_loss < best_val_loss:
#             best_val_loss     = val_loss
#             epochs_no_improve = 0
#             save_checkpoint(model, epoch, val_loss, checkpoint_path)
#             print(f"  💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
#         else:
#             epochs_no_improve += 1
#             print(f"  ⏳ Pas d'amélioration ({epochs_no_improve}/{PATIENCE})")

#         if epochs_no_improve >= PATIENCE:
#             print(f"\n🛑 Early stopping à l'epoch {epoch}.")
#             break

#     # ── 5. Évaluation finale ──────────────────────────────────────
#     print(f"\n{'═'*55}")
#     print("📊 Évaluation finale — meilleur modèle")
#     print(f"{'═'*55}")

#     if not Path(checkpoint_path).exists():
#         print("❌ Aucun checkpoint sauvegardé (toutes les losses étaient NaN).")
#         print("   → Vérifie tes tenseurs avec check_tensors.py")
#         return

#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
#     model.load_state_dict(checkpoint["model_state"])
#     print(f"   Chargé depuis epoch {checkpoint['epoch']}  "
#           f"(val_loss={checkpoint['metric']:.4f})\n")

#     _, all_targets, all_preds, all_probs = evaluate(model, test_loader, criterion, DEVICE)

#     print("📋 RAPPORT DE CLASSIFICATION")
#     print("─"*55)
#     print(classification_report(
#         all_targets, all_preds,
#         target_names=["Légitime (0)", "Fraude (1)"],
#         zero_division=0,
#     ))

#     cm = confusion_matrix(all_targets, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     print("🧩 MATRICE DE CONFUSION")
#     print("─"*55)
#     print(f"  Vrais Négatifs (Légitimes détectés) : {tn:>7,}")
#     print(f"  Faux Positifs  (Fausses alertes)    : {fp:>7,} ⚠️")
#     print(f"  Faux Négatifs  (Fraudes manquées)   : {fn:>7,} 🚨")
#     print(f"  Vrais Positifs (Fraudes détectées)  : {tp:>7,}")

#     print("\n📈 MÉTRIQUES DE PERFORMANCE")
#     print("─"*55)
#     try:
#         from sklearn.metrics import roc_curve, f1_score
#         roc_auc = roc_auc_score(all_targets, all_probs)
#         pr_auc  = average_precision_score(all_targets, all_probs)
#         print(f"  🌟 AUC-ROC                   : {roc_auc:.4f}")
#         print(f"  🎯 PR-AUC (Precision-Recall) : {pr_auc:.4f}")

#         # Recherche du seuil optimal (max F1 sur la fraude)
#         thresholds_to_try = np.arange(0.05, 0.95, 0.01)
#         best_f1, best_thresh = 0.0, 0.5
#         for t in thresholds_to_try:
#             preds_t = (all_probs >= t).astype(float)
#             f1 = f1_score(all_targets, preds_t, pos_label=1, zero_division=0)
#             if f1 > best_f1:
#                 best_f1, best_thresh = f1, t

#         print(f"\n  🔍 SEUIL OPTIMAL (max F1-fraude)")
#         print(f"  Seuil  : {best_thresh:.2f}  (au lieu de 0.50)")
#         print(f"  F1-fraude : {best_f1:.4f}")

#         preds_opt = (all_probs >= best_thresh).astype(float)
#         cm_opt = confusion_matrix(all_targets, preds_opt)
#         tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
#         print(f"\n  Avec seuil optimal :")
#         print(f"  Vrais Positifs (Fraudes détectées) : {tp_o:>6,}  / {int(all_targets.sum())}")
#         print(f"  Faux Positifs  (Fausses alertes)   : {fp_o:>6,}")
#         print(f"  Faux Négatifs  (Fraudes manquées)  : {fn_o:>6,}")
#         print(f"\n  Rappel fraude  : {tp_o / max(tp_o+fn_o,1):.2%}")
#         print(f"  Précision fraude: {tp_o / max(tp_o+fp_o,1):.2%}")

#     except ValueError as e:
#         print(f"  ⚠️ Calcul AUC impossible : {e}")

#     print(f"\n{'═'*55}\n")


# if __name__ == "__main__":
#     train_and_evaluate()









# """
# train_local.py — Entraînement local GRU + Approche A
# ══════════════════════════════════════════════════════
# Fixes v2 :
#   - API torch.amp (non dépréciée) pour GradScaler et autocast
#   - Gradient clipping (max_norm=1.0) → évite les explosions → nan
#   - pos_weight en FP32 forcé → évite overflow FP16
#   - Détection NaN dans les données au démarrage
#   - Checkpoint protégé : sauvegarde dès epoch 1 garantie
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from pathlib import Path
# from sklearn.metrics import (
#     classification_report,
#     roc_auc_score,
#     confusion_matrix,
#     average_precision_score,
# )

# from models.fraud_rnn import FraudRNN
# from data.dataloader import get_dataloaders_approach_A

# # ══════════════════════════════════════════════════════
# # ⚙️  CONFIGURATION
# # ══════════════════════════════════════════════════════

# DATA_DIR       = "data/node_1/tensors/"
# CHECKPOINT_DIR = "checkpoints/"

# # Modèle
# INPUT_DIM    = 26
# HIDDEN_DIM   = 128
# NUM_LAYERS   = 2
# DROPOUT_RATE = 0.2

# # Entraînement
# # EPOCHS        = 10
# EPOCHS        = 20
# BATCH_SIZE    = 256
# LEARNING_RATE = 0.0005
# SEQ_LEN       = 5
# TRAIN_RATIO   = 0.8
# POS_WEIGHT    = 167.0   # ratio réel : 362 974 légit / 2 174 fraudes
# # PATIENCE      = 3       # early stopping
# PATIENCE  = 5

# # ══════════════════════════════════════════════════════

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# USE_AMP = DEVICE.type == "cuda"   # Mixed Precision uniquement sur GPU

# if DEVICE.type == "cuda":
#     torch.backends.cudnn.benchmark = True
#     GPU_NAME = torch.cuda.get_device_name(0)
#     GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
# else:
#     GPU_NAME, GPU_MEM = "N/A", 0.0


# # ══════════════════════════════════════════════════════
# # 🔍 VÉRIFICATION DES TENSEURS
# # ══════════════════════════════════════════════════════

# def check_tensors(data_dir: str) -> bool:
#     """Vérifie l'absence de NaN/Inf dans les tenseurs avant l'entraînement."""
#     files = sorted(Path(data_dir).glob("X_batch_*.npy"))
#     total_nan = total_inf = 0

#     for f in files:
#         X = np.load(f)
#         total_nan += np.isnan(X).sum()
#         total_inf += np.isinf(X).sum()

#     if total_nan > 0 or total_inf > 0:
#         print(f"❌ Tenseurs corrompus → NaN:{total_nan}  Inf:{total_inf}")
#         print("   Relance generate_tensors_from_csv.py pour régénérer les tenseurs.")
#         return False

#     print(f"✅ Tenseurs vérifiés ({len(files)} fichiers) — aucun NaN/Inf")
#     return True


# # ══════════════════════════════════════════════════════
# # 🔧 UTILS
# # ══════════════════════════════════════════════════════

# def save_checkpoint(model, epoch, metric_val, path):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "epoch"      : epoch,
#         "model_state": model.state_dict(),
#         "metric"     : metric_val,
#     }, path)

# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss  = 0.0
#     all_preds, all_targets, all_probs = [], [], []

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             # AMP désactivé en évaluation :
#             # pos_weight=167 en FP16 → overflow → NaN dans la loss
#             logits = model(x).float()   # forcé FP32
#             loss   = criterion(logits, y.float())

#             probs = torch.sigmoid(logits)
#             preds = (probs > 0.5).float()

#             total_loss  += loss.item() * y.size(0)
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(y.cpu().numpy())

#     n = len(all_targets)
#     return (
#         total_loss / n if n > 0 else 0.0,
#         np.array(all_targets),
#         np.array(all_preds),
#         np.array(all_probs),
#     )


# # ══════════════════════════════════════════════════════
# # 🚀 TRAIN
# # ══════════════════════════════════════════════════════

# def train_and_evaluate():
#     print(f"\n{'═'*55}")
#     print(f"  🚀  Entraînement GRU — Approche A")
#     print(f"{'═'*55}")
#     print(f"  Device  : {DEVICE}  ({GPU_NAME}  {GPU_MEM:.1f} GB)" if DEVICE.type == "cuda"
#           else f"  Device  : CPU")
#     print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
#     print(f"  GRU     : hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  dropout={DROPOUT_RATE}")
#     print(f"  pos_weight : {POS_WEIGHT}  |  Mixed Precision : {USE_AMP}")
#     print(f"{'─'*55}\n")

#     # ── Vérification des tenseurs ─────────────────────────────────
#     if not check_tensors(DATA_DIR):
#         return

#     # ── 1. Données ────────────────────────────────────────────────
#     print("\n📂 Chargement (Approche A — Split Temporel Intra-User)...")
#     train_loader, test_loader = get_dataloaders_approach_A(
#         data_dir    = DATA_DIR,
#         train_ratio = TRAIN_RATIO,
#         batch_size  = BATCH_SIZE,
#         seq_len     = SEQ_LEN,
#     )

#     # ── 2. Modèle ─────────────────────────────────────────────────
#     model = FraudRNN(
#         input_dim    = INPUT_DIM,
#         hidden_dim   = HIDDEN_DIM,
#         num_layers   = NUM_LAYERS,
#         dropout_rate = DROPOUT_RATE,
#     ).to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n🧠 Paramètres entraînables : {total_params:,}")

#     # ── 3. Loss / Optimizer / Scheduler ──────────────────────────
#     # FIX : pos_weight en float() explicite → reste FP32 même avec AMP
#     pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
#     criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#     # optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

#     scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=2
#     )

#     # ✅ API non dépréciée : torch.amp.GradScaler
#     scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

#     # ── 4. Boucle d'entraînement ──────────────────────────────────
#     best_val_loss     = float("inf")
#     epochs_no_improve = 0
#     checkpoint_path   = f"{CHECKPOINT_DIR}best_gru_approachA.pt"

#     print(f"\n{'─'*55}")

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         total_train_loss = 0.0
#         n_train          = 0
#         has_nan          = False

#         for x, y in train_loader:
#             x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             # Forward en FP16 (vitesse), loss en FP32 (stabilité)
#             with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
#                 logits = model(x)
#             logits_fp32 = logits.float()   # repasse en FP32 avant la loss
#             loss = criterion(logits_fp32, y.float())

#             # Détection NaN dans la loss
#             if torch.isnan(loss):
#                 has_nan = True
#                 continue

#             loss.backward()   # pas de scaler car loss est en FP32

#             # Gradient clipping → évite les explosions
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()

#             total_train_loss += loss.item() * y.size(0)
#             n_train          += y.size(0)

#         if has_nan:
#             print(f"⚠️  Epoch {epoch} : NaN détectés dans la loss → batches ignorés")

#         avg_train_loss = total_train_loss / n_train if n_train > 0 else float("nan")
#         val_loss, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)

#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]["lr"]

#         print(f"Epoch [{epoch:>2}/{EPOCHS}] "
#               f"| Train Loss: {avg_train_loss:.4f} "
#               f"| Val Loss: {val_loss:.4f} "
#               f"| LR: {current_lr:.6f}")

#         # Checkpoint — sauvegarde si meilleure val_loss ET loss valide
#         if not np.isnan(val_loss) and val_loss < best_val_loss:
#             best_val_loss     = val_loss
#             epochs_no_improve = 0
#             save_checkpoint(model, epoch, val_loss, checkpoint_path)
#             print(f"  💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
#         else:
#             epochs_no_improve += 1
#             print(f"  ⏳ Pas d'amélioration ({epochs_no_improve}/{PATIENCE})")

#         if epochs_no_improve >= PATIENCE:
#             print(f"\n🛑 Early stopping à l'epoch {epoch}.")
#             break

#     # ── 5. Évaluation finale ──────────────────────────────────────
#     print(f"\n{'═'*55}")
#     print("📊 Évaluation finale — meilleur modèle")
#     print(f"{'═'*55}")

#     if not Path(checkpoint_path).exists():
#         print("❌ Aucun checkpoint sauvegardé (toutes les losses étaient NaN).")
#         print("   → Vérifie tes tenseurs avec check_tensors.py")
#         return

#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
#     model.load_state_dict(checkpoint["model_state"])
#     print(f"   Chargé depuis epoch {checkpoint['epoch']}  "
#           f"(val_loss={checkpoint['metric']:.4f})\n")

#     _, all_targets, all_preds, all_probs = evaluate(model, test_loader, criterion, DEVICE)

#     print("📋 RAPPORT DE CLASSIFICATION")
#     print("─"*55)
#     print(classification_report(
#         all_targets, all_preds,
#         target_names=["Légitime (0)", "Fraude (1)"],
#         zero_division=0,
#     ))

#     cm = confusion_matrix(all_targets, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     print("🧩 MATRICE DE CONFUSION")
#     print("─"*55)
#     print(f"  Vrais Négatifs (Légitimes détectés) : {tn:>7,}")
#     print(f"  Faux Positifs  (Fausses alertes)    : {fp:>7,} ⚠️")
#     print(f"  Faux Négatifs  (Fraudes manquées)   : {fn:>7,} 🚨")
#     print(f"  Vrais Positifs (Fraudes détectées)  : {tp:>7,}")

#     print("\n📈 MÉTRIQUES DE PERFORMANCE")
#     print("─"*55)
#     try:
#         from sklearn.metrics import roc_curve, f1_score
#         roc_auc = roc_auc_score(all_targets, all_probs)
#         pr_auc  = average_precision_score(all_targets, all_probs)
#         print(f"  🌟 AUC-ROC                   : {roc_auc:.4f}")
#         print(f"  🎯 PR-AUC (Precision-Recall) : {pr_auc:.4f}")

#         # Recherche du seuil optimal (max F1 sur la fraude)
#         thresholds_to_try = np.arange(0.05, 0.95, 0.01)
#         best_f1, best_thresh = 0.0, 0.5
#         for t in thresholds_to_try:
#             preds_t = (all_probs >= t).astype(float)
#             f1 = f1_score(all_targets, preds_t, pos_label=1, zero_division=0)
#             if f1 > best_f1:
#                 best_f1, best_thresh = f1, t

#         print(f"\n  🔍 SEUIL OPTIMAL (max F1-fraude)")
#         print(f"  Seuil  : {best_thresh:.2f}  (au lieu de 0.50)")
#         print(f"  F1-fraude : {best_f1:.4f}")

#         preds_opt = (all_probs >= best_thresh).astype(float)
#         cm_opt = confusion_matrix(all_targets, preds_opt)
#         tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
#         print(f"\n  Avec seuil optimal :")
#         print(f"  Vrais Positifs (Fraudes détectées) : {tp_o:>6,}  / {int(all_targets.sum())}")
#         print(f"  Faux Positifs  (Fausses alertes)   : {fp_o:>6,}")
#         print(f"  Faux Négatifs  (Fraudes manquées)  : {fn_o:>6,}")
#         print(f"\n  Rappel fraude  : {tp_o / max(tp_o+fn_o,1):.2%}")
#         print(f"  Précision fraude: {tp_o / max(tp_o+fp_o,1):.2%}")

#     except ValueError as e:
#         print(f"  ⚠️ Calcul AUC impossible : {e}")

#     print(f"\n{'═'*55}\n")


# if __name__ == "__main__":
#     train_and_evaluate()














# """
# train_local.py — Entraînement local GRU + Approche A
# ══════════════════════════════════════════════════════
# Optimisations GPU :
#   - torch.cuda.amp  : Mixed Precision (FP16) → x1.5 à x2 plus rapide sur RTX/Quadro
#   - cudnn.benchmark : Sélection automatique du kernel CUDA optimal
#   - non_blocking    : Transferts CPU→GPU asynchrones
#   - pos_weight=167  : Calculé depuis le vrai ratio (2 174 fraudes / 362 974 légitimes)
#   - LR Scheduler    : ReduceLROnPlateau → adapte le lr si le modèle stagne
#   - Early Stopping  : Arrêt automatique si pas d'amélioration pendant N epochs
#   - Checkpointing   : Sauvegarde automatique du meilleur modèle
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
# import numpy as np
# from sklearn.metrics import (
#     classification_report,
#     roc_auc_score,
#     confusion_matrix,
#     average_precision_score,
# )

# from models.fraud_rnn import FraudRNN
# from data.dataloader import get_dataloaders_approach_A

# # ══════════════════════════════════════════════════════
# # ⚙️  CONFIGURATION — modifie ici pour tes expériences
# # ══════════════════════════════════════════════════════

# DATA_DIR       = "data/node_1/tensors/"
# CHECKPOINT_DIR = "checkpoints/"

# # Modèle
# INPUT_DIM      = 26
# HIDDEN_DIM     = 128
# NUM_LAYERS     = 2       # couches GRU empilées
# DROPOUT_RATE   = 0.2

# # Entraînement
# EPOCHS         = 10
# BATCH_SIZE     = 256     # plus grand batch → meilleure utilisation du GPU
# LEARNING_RATE  = 0.001
# SEQ_LEN        = 5
# TRAIN_RATIO    = 0.8

# # Classe déséquilibrée : 167 légitimes pour 1 fraude
# # pos_weight dit au modèle "rater une fraude coûte 167x plus cher"
# POS_WEIGHT     = 167.0

# # Early stopping
# PATIENCE       = 3       # arrêt si pas d'amélioration pendant 3 epochs

# # ══════════════════════════════════════════════════════════

# # ── GPU setup ─────────────────────────────────────────────
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if DEVICE.type == "cuda":
#     # Sélectionne automatiquement le kernel CUDA le plus rapide pour ta carte
#     torch.backends.cudnn.benchmark = True
#     GPU_NAME = torch.cuda.get_device_name(0)
#     GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
# else:
#     GPU_NAME = "N/A"
#     GPU_MEM  = 0.0


# # ══════════════════════════════════════════════════════════
# # 🔧 UTILS
# # ══════════════════════════════════════════════════════════

# import os
# from pathlib import Path

# def save_checkpoint(model, epoch, metric_val, path):
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         "epoch"     : epoch,
#         "model_state": model.state_dict(),
#         "metric"    : metric_val,
#     }, path)

# def evaluate(model, loader, criterion, device):
#     """Passe complète sur le loader → loss, preds, targets, probs."""
#     model.eval()
#     total_loss  = 0.0
#     all_preds   = []
#     all_targets = []
#     all_probs   = []

#     with torch.no_grad():
#         for x, y in loader:
#             # non_blocking=True : le CPU prépare le prochain batch pendant que le GPU calcule
#             x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

#             with autocast(enabled=(device.type == "cuda")):
#                 logits = model(x)
#                 loss   = criterion(logits, y)

#             probs = torch.sigmoid(logits)
#             preds = (probs > 0.5).float()

#             total_loss  += loss.item() * y.size(0)
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(y.cpu().numpy())

#     n        = len(all_targets)
#     avg_loss = total_loss / n if n > 0 else 0.0
#     return avg_loss, np.array(all_targets), np.array(all_preds), np.array(all_probs)


# # ══════════════════════════════════════════════════════════
# # 🚀 TRAIN
# # ══════════════════════════════════════════════════════════

# def train_and_evaluate():
#     print(f"\n{'═'*55}")
#     print(f"  🚀  Entraînement GRU — Approche A")
#     print(f"{'═'*55}")
#     print(f"  Device  : {DEVICE}  ({GPU_NAME}  {GPU_MEM:.1f} GB)" if DEVICE.type == "cuda"
#           else f"  Device  : {DEVICE} (pas de GPU détecté)")
#     print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
#     print(f"  GRU     : input={INPUT_DIM}  hidden={HIDDEN_DIM}  layers={NUM_LAYERS}")
#     print(f"  pos_weight : {POS_WEIGHT}  (ratio réel fraude/légit)")
#     print(f"{'─'*55}\n")

#     # ── 1. Données ────────────────────────────────────────────────
#     print("📂 Chargement des données (Approche A — Split Temporel Intra-User)...")
#     train_loader, test_loader = get_dataloaders_approach_A(
#         data_dir    = DATA_DIR,
#         train_ratio = TRAIN_RATIO,
#         batch_size  = BATCH_SIZE,
#         seq_len     = SEQ_LEN,
#     )

#     # ── 2. Modèle ─────────────────────────────────────────────────
#     model = FraudRNN(
#         input_dim    = INPUT_DIM,
#         hidden_dim   = HIDDEN_DIM,
#         num_layers   = NUM_LAYERS,
#         dropout_rate = DROPOUT_RATE,
#     ).to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\n🧠 Paramètres entraînables : {total_params:,}")

#     # ── 3. Loss / Optimizer / Scheduler ──────────────────────────
#     pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE)
#     criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # Réduit le LR de moitié si la val_loss ne s'améliore pas pendant 2 epochs
#     scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=2, verbose=True
#     )

#     # Mixed Precision : calculs en FP16 sur GPU → plus rapide, moins de VRAM
#     scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

#     # ── 4. Boucle d'entraînement ──────────────────────────────────
#     best_val_loss    = float("inf")
#     epochs_no_improve = 0
#     checkpoint_path  = f"{CHECKPOINT_DIR}best_gru_approachA.pt"

#     print(f"\n{'─'*55}")
#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         total_train_loss = 0.0
#         n_train          = 0

#         for x, y in train_loader:
#             x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)  # légèrement plus rapide que zero_grad()

#             # Forward en FP16 (Mixed Precision)
#             with autocast(enabled=(DEVICE.type == "cuda")):
#                 logits = model(x)
#                 loss   = criterion(logits, y)

#             # Backward avec scaler (évite les underflows FP16)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             total_train_loss += loss.item() * y.size(0)
#             n_train          += y.size(0)

#         avg_train_loss = total_train_loss / n_train

#         # Validation rapide
#         val_loss, _, _, _ = evaluate(model, test_loader, criterion, DEVICE)

#         # Scheduler
#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]["lr"]

#         print(f"Epoch [{epoch:>2}/{EPOCHS}] "
#               f"| Train Loss: {avg_train_loss:.4f} "
#               f"| Val Loss: {val_loss:.4f} "
#               f"| LR: {current_lr:.6f}")

#         # Checkpointing — sauvegarde le meilleur modèle
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#             save_checkpoint(model, epoch, val_loss, checkpoint_path)
#             print(f"  💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
#         else:
#             epochs_no_improve += 1
#             print(f"  ⏳ Pas d'amélioration ({epochs_no_improve}/{PATIENCE})")

#         # Early stopping
#         if epochs_no_improve >= PATIENCE:
#             print(f"\n🛑 Early stopping déclenché à l'epoch {epoch}.")
#             break

#     # ── 5. Évaluation finale avec le MEILLEUR modèle ─────────────
#     print(f"\n{'═'*55}")
#     print("📊 Évaluation finale — meilleur modèle chargé")
#     print(f"{'═'*55}")

#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#     model.load_state_dict(checkpoint["model_state"])
#     print(f"   Chargé depuis epoch {checkpoint['epoch']}  (val_loss={checkpoint['metric']:.4f})\n")

#     _, all_targets, all_preds, all_probs = evaluate(model, test_loader, criterion, DEVICE)

#     # Rapport de classification
#     print("📋 RAPPORT DE CLASSIFICATION")
#     print("─"*55)
#     print(classification_report(
#         all_targets, all_preds,
#         target_names=["Légitime (0)", "Fraude (1)"],
#         zero_division=0,
#     ))

#     # Matrice de confusion
#     cm = confusion_matrix(all_targets, all_preds)
#     tn, fp, fn, tp = cm.ravel()
#     print("🧩 MATRICE DE CONFUSION")
#     print("─"*55)
#     print(f"  Vrais Négatifs (Légitimes détectés) : {tn:>7,}")
#     print(f"  Faux Positifs  (Fausses alertes)    : {fp:>7,} ⚠️")
#     print(f"  Faux Négatifs  (Fraudes manquées)   : {fn:>7,} 🚨")
#     print(f"  Vrais Positifs (Fraudes détectées)  : {tp:>7,}")

#     # AUC
#     print("\n📈 MÉTRIQUES DE PERFORMANCE")
#     print("─"*55)
#     try:
#         roc_auc = roc_auc_score(all_targets, all_probs)
#         pr_auc  = average_precision_score(all_targets, all_probs)
#         print(f"  🌟 AUC-ROC                  : {roc_auc:.4f}")
#         print(f"  🎯 PR-AUC (Precision-Recall): {pr_auc:.4f}")
#     except ValueError:
#         print("  ⚠️ Calcul d'AUC impossible (manque de classes).")

#     print(f"\n{'═'*55}\n")


# if __name__ == "__main__":
#     train_and_evaluate()






# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# # from sklearn.metrics import classification_report, roc_auc_score

# from sklearn.metrics import (
#     classification_report, 
#     roc_auc_score, 
#     confusion_matrix, 
#     average_precision_score
# )

# # Import de tes modules
# from models.fraud_rnn import FraudRNN
# # from data.dataloader import get_dataloader
# from data.dataloader import get_dataloaders_approach_A

# # Configuration
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EPOCHS = 10
# BATCH_SIZE = 64
# LEARNING_RATE = 0.001

# def train_and_evaluate():
#     print(f"🚀 Démarrage de l'entraînement local sur : {DEVICE}")

#     # 1. Chargement des données (Assure-toi que ton dataloader pointe vers les bons dossiers)
# #     print("📂 Chargement des données...")
# # # Pour l'entraînement (il va trouver tous les X_*.npy dans ce dossier)
# #     train_loader = get_dataloader(
# #         data_dir="data/local_sample/train/", 
# #         x_prefix="X_", 
# #         y_prefix="y_",
# #         batch_size=BATCH_SIZE, 
# #         seq_len=5, 
# #         shuffle=True
# #     )
    
# #     test_loader = get_dataloader(
# #         data_dir="data/local_sample/test/", 
# #         x_prefix="X_", 
# #         y_prefix="y_",
# #         batch_size=BATCH_SIZE, 
# #         seq_len=5, 
# #         shuffle=False
# #     )



#     print("📂 Chargement des données (Approche A - Split Temporel Intra-User)...")
    
#     # Le Dataloader s'occupe de tout (Scanner, concaténer, et séparer 80/20 par utilisateur)
#     data_dir = "data/node_1/tensors/"
#     # train_loader, test_loader = get_dataloaders_approach_A(
#     #     data_dir=data_dir, 
#     #     train_ratio=0.8, 
#     #     batch_size=BATCH_SIZE, 
#     #     seq_len=5
#     # )
#     train_loader, test_loader = get_dataloaders_approach_A(
#     data_dir=data_dir, 
#     train_ratio=0.8, 
#     batch_size=BATCH_SIZE, 
#     seq_len=5
# )

#     # 2. Initialisation du Modèle
#     model = FraudRNN(input_dim=26, hidden_dim=128, num_layers=2, dropout_rate=0.2).to(DEVICE)
    
#     # BCEWithLogitsLoss gère le déséquilibre de classe si on lui passe le paramètre pos_weight (optionnel)
#     # criterion = nn.BCEWithLogitsLoss()
#     pos_weight = torch.tensor([50.0]).to(DEVICE)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     # 3. Boucle d'entraînement
#     for epoch in range(EPOCHS):
#         model.train()
#         total_train_loss = 0.0
        
#         for x, y in train_loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
            
#             optimizer.zero_grad()
#             logits = model(x)
            
#             loss = criterion(logits, y)
#             loss.backward()
#             optimizer.step()
            
#             total_train_loss += loss.item()
            
#         avg_train_loss = total_train_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f}")

#     # # 4. Évaluation (Validation)
#     # print("\n📊 Évaluation finale sur le set de test...")
#     # model.eval()
#     # all_preds = []
#     # all_targets = []
#     # all_probs = []

#     # with torch.no_grad():
#     #     for x, y in test_loader:
#     #         x, y = x.to(DEVICE), y.to(DEVICE)
            
#     #         logits = model(x)
#     #         probs = torch.sigmoid(logits)
#     #         preds = (probs > 0.5).float()
            
#     #         all_probs.extend(probs.cpu().numpy())
#     #         all_preds.extend(preds.cpu().numpy())
#     #         all_targets.extend(y.cpu().numpy())

#     # # 5. Calcul des métriques de Fraude
#     # all_targets = np.array(all_targets)
#     # all_preds = np.array(all_preds)
#     # all_probs = np.array(all_probs)

#     # print("\n=== RAPPORT DE CLASSIFICATION ===")
#     # print(classification_report(all_targets, all_preds, target_names=["Légitime (0)", "Fraude (1)"]))
    
#     # try:
#     #     auc = roc_auc_score(all_targets, all_probs)
#     #     print(f"🌟 AUC-ROC : {auc:.4f}")
#     # except ValueError:
#     #     print("⚠️ Pas assez de classes différentes dans le set de test pour calculer l'AUC.")

#     print("\n📊 Évaluation finale sur le set de test...")
#     model.eval()
#     all_preds = []
#     all_targets = []
#     all_probs = []

#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             logits = model(x)
#             probs = torch.sigmoid(logits)
#             preds = (probs > 0.5).float()
            
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(y.cpu().numpy())

#     # 5. Calcul des métriques avancées
#     all_targets = np.array(all_targets)
#     all_preds = np.array(all_preds)
#     all_probs = np.array(all_probs)

#     # Métriques standards
#     print("\n" + "="*30)
#     print("📋 RAPPORT DE CLASSIFICATION")
#     print("="*30)
#     print(classification_report(all_targets, all_preds, target_names=["Légitime (0)", "Fraude (1)"]))

#     # Matrice de Confusion
#     cm = confusion_matrix(all_targets, all_preds)
#     tn, fp, fn, tp = cm.ravel()
    
#     print("\n" + "="*30)
#     print("🧩 MATRICE DE CONFUSION")
#     print("="*30)
#     print(f"Vrais Négatifs (Légitimes détectés) : {tn}")
#     print(f"Faux Positifs (Fausses alertes)    : {fp} ⚠️")
#     print(f"Faux Négatifs (Fraudes manquées)   : {fn} 🚨")
#     print(f"Vrais Positifs (Fraudes détectées) : {tp}")

#     # Scores de courbes (AUC)
#     print("\n" + "="*30)
#     print("📈 MÉTRIQUES DE PERFORMANCE")
#     print("="*30)
    
#     try:
#         roc_auc = roc_auc_score(all_targets, all_probs)
#         pr_auc = average_precision_score(all_targets, all_probs)
#         print(f"🌟 AUC-ROC : {roc_auc:.4f}")
#         print(f"🎯 PR-AUC (Precision-Recall) : {pr_auc:.4f}")
#     except ValueError:
#         print("⚠️ Calcul d'AUC impossible (manque de classes).")

#     print("\n" + "="*30)

# if __name__ == "__main__":
#     train_and_evaluate()




# # import torch

# # print("CUDA est-il disponible ? :", torch.cuda.is_available())

# # if torch.cuda.is_available():
# #     print("Nom de la carte graphique :", torch.cuda.get_device_name(0))
# #     print("Nombre de GPU détectés :", torch.cuda.device_count())
# # else:
# #     print("Attention : PyTorch ne voit que le CPU.")