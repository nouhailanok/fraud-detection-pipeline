# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# from pathlib import Path

# class FraudSequenceDataset(Dataset):
#     """
#     Dataset PyTorch capable de scanner un dossier de batchs npy.
#     Structure attendue de X : [PAN_HASH (colonne 0), feature1, ..., feature26]
#     Structure attendue de y : [is_fraud]
#     """
#     def __init__(self, data_dir, sequence_length=5):
#         self.data_dir = Path(data_dir)
#         self.sequence_length = sequence_length
        
#         # 1. Scan et chargement de tous les batchs disponibles
#         # sorted() est CRUCIAL pour garder l'ordre des timestamps Kafka
#         x_files = sorted(list(self.data_dir.glob("X_batch_*.npy")))
#         y_files = sorted(list(self.data_dir.glob("y_batch_*.npy")))
        
#         if not x_files:
#             raise FileNotFoundError(f"❌ Aucun fichier .npy trouvé dans {self.data_dir}")
            
#         # Chargement et concaténation de tous les batchs
#         all_x = []
#         all_y = []
#         for x_f, y_f in zip(x_files, y_files):
#             all_x.append(np.load(x_f))
#             all_y.append(np.load(y_f))
            
#         raw_X = np.vstack(all_x).astype(np.float32)
#         raw_y = np.vstack(all_y).astype(np.float32)
        
#         # 2. Séparation ID (colonne 0) et Features (colonnes 1-26)
#         self.user_ids = raw_X[:, 0] 
#         self.features = raw_X[:, 1:] 
#         self.y = raw_y.flatten()

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         """
#         Construit une séquence temporelle avec PADDING par utilisateur.
#         L'ordre temporel est préservé à l'intérieur de chaque user.
#         """
#         current_user = self.user_ids[idx]
#         sequence = []
        
#         for i in range(self.sequence_length - 1, -1, -1):
#             target_idx = idx - i
            
#             # Si on dépasse le début ou si l'utilisateur change -> Padding Zéro
#             if target_idx < 0 or self.user_ids[target_idx] != current_user:
#                 sequence.append(np.zeros(self.features.shape[1]))
#             else:
#                 sequence.append(self.features[target_idx])
        
#         x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
#         y_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
        
#         return x_tensor, y_tensor


# def get_split_dataloaders(data_dir, train_ratio=0.8, batch_size=64, seq_len=5, random_seed=42):
#     """
#     Divise les données du nœud en Train/Test par UTILISATEUR.

#     - 80% des users  → Train  (toutes leurs transactions, ordre temporel préservé)
#     - 20% des users  → Test   (users jamais vus en train → zéro data leakage)

#     L'ordre temporel des transactions est préservé à l'intérieur de chaque user
#     grâce au sorted() lors du chargement des batchs Kafka.
#     """
#     full_dataset = FraudSequenceDataset(data_dir, sequence_length=seq_len)

#     # ------------------------------------------------------------------ #
#     # 1. Split des USERS (pas des transactions)                           #
#     # ------------------------------------------------------------------ #
#     unique_users = np.unique(full_dataset.user_ids)

#     # Shuffle reproductible des users
#     rng = np.random.default_rng(random_seed)
#     shuffled_users = rng.permutation(unique_users)

#     split_point      = int(len(shuffled_users) * train_ratio)
#     train_users      = set(shuffled_users[:split_point])   # 80% des users
#     test_users       = set(shuffled_users[split_point:])   # 20% des users

#     # ------------------------------------------------------------------ #
#     # 2. Collecte des indices de transactions par groupe                  #
#     #    L'ordre est déjà chronologique grâce au sorted() du Dataset     #
#     # ------------------------------------------------------------------ #
#     train_indices = []
#     test_indices  = []

#     for idx, user in enumerate(full_dataset.user_ids):
#         if user in train_users:
#             train_indices.append(idx)
#         else:
#             test_indices.append(idx)

#     # ------------------------------------------------------------------ #
#     # 3. Création des Subsets PyTorch                                     #
#     # ------------------------------------------------------------------ #
#     train_ds = Subset(full_dataset, train_indices)
#     test_ds  = Subset(full_dataset, test_indices)

#     # ------------------------------------------------------------------ #
#     # 4. DataLoaders                                                      #
#     #    - Train : shuffle=True  → mélange les transactions inter-users  #
#     #              (la séquence est auto-portée dans __getitem__)         #
#     #    - Test  : shuffle=False → évaluation stable et reproductible    #
#     # ------------------------------------------------------------------ #
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,   # OK : chaque item porte sa propre séquence
#         num_workers=0,
#         pin_memory=True
#     )

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True
#     )

#     # ------------------------------------------------------------------ #
#     # 5. Logs de vérification                                             #
#     # ------------------------------------------------------------------ #
#     fraud_train = sum(full_dataset.y[i] for i in train_indices)
#     fraud_test  = sum(full_dataset.y[i] for i in test_indices)

#     print(f"✅ Split par users")
#     print(f"   Users  → Train : {len(train_users)} | Test : {len(test_users)}")
#     print(f"   Transactions → Train : {len(train_indices)} | Test : {len(test_indices)}")
#     print(f"   Fraudes → Train : {int(fraud_train)} ({100*fraud_train/len(train_indices):.2f}%)"
#           f" | Test : {int(fraud_test)} ({100*fraud_test/len(test_indices):.2f}%)")

#     return train_loader, test_loader


# # --- ZONE DE TEST ---
# if __name__ == "__main__":
#     # Exemple d'usage :
#     # path = "data/node_1/tensors"
#     # tr_loader, te_loader = get_split_dataloaders(path, train_ratio=0.8)
#     # print(f"Batches Train : {len(tr_loader)} | Batches Test : {len(te_loader)}")
#     pass








# """
# dataloader.py — Pipeline de chargement de données pour la détection de fraude (FL)
# ====================================================================================
# Stratégies de split disponibles :
#   - Approche A : Split temporel INTRA-utilisateur  (80% passé / 20% futur par user)
#   - Approche B : Split par POPULATION              (80% users connus / 20% users inconnus)

# Contrainte commune :
#   - Séquences de longueur SEQ_LEN=5, jamais de mélange inter-utilisateurs.
#   - Colonne 0 des .npy = PAN_ID (utilisée pour les frontières de séquence, retirée
#     avant l'entrée dans le modèle GRU).

# Normalisation :
#   - StandardScaler fitté UNIQUEMENT sur les indices train (pas de data leakage).
#   - Appliqué à l'ensemble du tableau features avant __getitem__.
#   - Résout le problème NaN causé par Time_Delta_Sec (~31M) et Velocity_Sum_24H (~2.7M)
#     qui dépassent le max FP16 (65 504) sans normalisation.
# """

# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# from pathlib import Path
# from collections import defaultdict


# # ============================================================================
# # 📦 DATASET — Chargement, tri et séquençage
# # ============================================================================

# class FraudSequenceDataset(Dataset):
#     """
#     Charge tous les fichiers X_batch_*.npy / y_batch_*.npy d'un dossier,
#     les concatène et les TRIE PAR UTILISATEUR (tri stable → ordre temporel
#     intra-user préservé).

#     Structure attendue des .npy
#     ───────────────────────────
#     X  : shape (N, 27)  →  col 0 = PAN_ID  |  col 1-26 = 26 features métier
#     y  : shape (N, 1)   →  label binaire (0=légitime, 1=fraude)

#     Sortie de __getitem__
#     ─────────────────────
#     x_tensor : (SEQ_LEN, 26)  — séquence glissante, zéro-paddée aux frontières
#     y_tensor : (1,)            — label de la transaction courante
#     """

#     def __init__(self, data_dir: str | Path, sequence_length: int = 5):
#         self.data_dir        = Path(data_dir)
#         self.sequence_length = sequence_length
#         self._scaler_fitted  = False
#         self._feat_mean      = None
#         self._feat_std       = None

#         # ── 1. Scan et chargement ────────────────────────────────────────────
#         x_files = sorted(self.data_dir.glob("X_batch_*.npy"))
#         y_files = sorted(self.data_dir.glob("y_batch_*.npy"))

#         if not x_files:
#             raise FileNotFoundError(
#                 f"❌ Aucun fichier X_batch_*.npy trouvé dans {self.data_dir}"
#             )
#         if len(x_files) != len(y_files):
#             raise ValueError(
#                 f"❌ Nombre de fichiers X ({len(x_files)}) ≠ Y ({len(y_files)})"
#             )

#         all_x, all_y = [], []
#         for xf, yf in zip(x_files, y_files):
#             all_x.append(np.load(xf))
#             all_y.append(np.load(yf))

#         raw_X = np.vstack(all_x).astype(np.float32)
#         raw_y = np.vstack(all_y).astype(np.float32).flatten()

#         # ── 2. Tri stable par PAN_ID ─────────────────────────────────────────
#         sort_idx = np.argsort(raw_X[:, 0], kind="stable")
#         raw_X    = raw_X[sort_idx]
#         raw_y    = raw_y[sort_idx]

#         # ── 3. Séparation ID / Features ──────────────────────────────────────
#         self.user_ids = raw_X[:, 0]      # (N,)   identifiant PAN_HASH
#         self.features = raw_X[:, 1:]     # (N,26) features métier — pas encore normalisées
#         self.y        = raw_y            # (N,)

#         n_users = len(np.unique(self.user_ids))
#         print(
#             f"📦 Dataset chargé : {len(self.features):,} transactions"
#             f" | {n_users:,} utilisateurs uniques"
#             f" | {int(self.y.sum()):,} fraudes"
#             f" ({100 * self.y.mean():.2f} %)"
#         )

#     # ── Normalisation ─────────────────────────────────────────────────────────

#     def fit_and_apply_scaler(self, train_indices: list) -> None:
#         """
#         Fitte un StandardScaler sur les features des indices TRAIN uniquement,
#         puis l'applique à TOUT le tableau (train + test).

#         Pourquoi fit sur train seulement ?
#         → Évite le data leakage : le modèle ne doit pas "voir" la distribution
#           du test avant l'évaluation.

#         Résout le NaN FP16 :
#         → Time_Delta_Sec  : 31 000 000  → ~0 centré réduit
#         → Velocity_Sum_24H:  2 700 000  → ~0 centré réduit
#         """
#         train_features = self.features[train_indices]          # (N_train, 26)

#         # mean = train_features.mean(axis=0)                     # (26,)
#         # std  = train_features.std(axis=0)                      # (26,)
#         # std  = np.where(std == 0, 1.0, std)                    # évite division par zéro

#         median = np.median(train_features, axis=0)
#         q75, q25 = np.percentile(train_features, [75, 25], axis=0)
#         iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)

#         self._feat_mean     = median
#         self._feat_std      = iqr
#         self._scaler_fitted = True

#         # Application sur tout le dataset (in-place)
#         self.features = (self.features - median) / iqr
#         self.features = np.clip(self.features, -5.0, 5.0)

#         print(f"📐 Normalisation appliquée (fitté sur {len(train_indices):,} txns train)")
#         print(f"   Avant : min={train_features.min():.1f}  max={train_features.max():.1f}")
#         print(f"   Après : min={self.features.min():.3f}  max={self.features.max():.3f}")

#     # ── Dataset interface ─────────────────────────────────────────────────────

#     def __len__(self) -> int:
#         return len(self.features)

#     def __getitem__(self, idx: int):
#         """
#         Construit la séquence temporelle de longueur SEQ_LEN centrée sur idx.
#         Padding zéro aux frontières inter-utilisateurs.
#         Le modèle GRU ne reçoit jamais l'ID utilisateur (col 0 retirée).
#         """
#         current_user = self.user_ids[idx]
#         sequence     = []

#         for lag in range(self.sequence_length - 1, -1, -1):
#             target_idx = idx - lag
#             if target_idx < 0 or self.user_ids[target_idx] != current_user:
#                 sequence.append(np.zeros(self.features.shape[1], dtype=np.float32))
#             else:
#                 sequence.append(self.features[target_idx])

#         x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)  # (SEQ_LEN, 26)
#         y_tensor = torch.tensor([self.y[idx]],       dtype=torch.float32)  # (1,)

#         return x_tensor, y_tensor


# # ============================================================================
# # 🔧 UTILITAIRES INTERNES
# # ============================================================================

# def _build_loaders(
#     dataset:       FraudSequenceDataset,
#     train_indices: list,
#     test_indices:  list,
#     batch_size:    int,
# ) -> tuple[DataLoader, DataLoader]:
#     train_loader = DataLoader(
#         Subset(dataset, train_indices),
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False,
#     )
#     test_loader = DataLoader(
#         Subset(dataset, test_indices),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#         drop_last=False,
#     )
#     return train_loader, test_loader


# def _print_split_report(
#     dataset:       FraudSequenceDataset,
#     train_indices: list,
#     test_indices:  list,
#     approach_name: str,
#     extra_info:    str = "",
# ) -> None:
#     y        = dataset.y
#     n_train  = len(train_indices)
#     n_test   = len(test_indices)
#     fraud_tr = float(sum(y[i] for i in train_indices))
#     fraud_te = float(sum(y[i] for i in test_indices))

#     print(f"\n{'─' * 54}")
#     print(f"  📊  {approach_name}")
#     if extra_info:
#         print(f"  {extra_info}")
#     print(f"{'─' * 54}")
#     print(f"  Transactions │ Train : {n_train:>7,}  │  Test : {n_test:>7,}")
#     print(
#         f"  Fraudes      │ Train : {int(fraud_tr):>7,}"
#         f" ({100 * fraud_tr / max(n_train, 1):.2f} %)"
#         f"  │  Test : {int(fraud_te):>7,}"
#         f" ({100 * fraud_te / max(n_test, 1):.2f} %)"
#     )
#     print(f"{'─' * 54}\n")


# # ============================================================================
# # 🅰️  APPROCHE A — Split temporel INTRA-utilisateur
# # ============================================================================

# def get_dataloaders_approach_A(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
# ) -> tuple[DataLoader, DataLoader]:
#     """
#     Approche A — Split temporel intra-utilisateur
#     ═══════════════════════════════════════════════
#     Pour chaque utilisateur, les 80 % premières transactions → TRAIN,
#     les 20 % dernières → TEST.

#     Normalisation : StandardScaler fitté sur train uniquement.
#     """
#     dataset = FraudSequenceDataset(data_dir, sequence_length=seq_len)

#     # ── Split vectorisé O(N) ──────────────────────────────────────────────────
#     user_ids_arr  = dataset.user_ids
#     change_mask   = np.concatenate(([True], user_ids_arr[1:] != user_ids_arr[:-1]))
#     group_starts  = np.where(change_mask)[0]
#     group_ends    = np.concatenate((group_starts[1:], [len(user_ids_arr)]))
#     group_sizes   = group_ends - group_starts
#     split_pts     = np.maximum(1, (group_sizes * train_ratio).astype(int))

#     train_indices_list, test_indices_list = [], []
#     for start, size, sp in zip(group_starts, group_sizes, split_pts):
#         indices = np.arange(start, start + size)
#         train_indices_list.append(indices[:sp])
#         if sp < size:
#             test_indices_list.append(indices[sp:])

#     train_indices = np.concatenate(train_indices_list).tolist()
#     test_indices  = np.concatenate(test_indices_list).tolist() if test_indices_list else []

#     _print_split_report(
#         dataset, train_indices, test_indices,
#         "Approche A — Split Temporel Intra-User",
#         extra_info=f"Users : {len(group_starts):,}  │  Ratio : {int(train_ratio*100)}/{int((1-train_ratio)*100)}",
#     )

#     # ── Normalisation fittée sur train uniquement ─────────────────────────────
#     dataset.fit_and_apply_scaler(train_indices)

#     return _build_loaders(dataset, train_indices, test_indices, batch_size)


# # ============================================================================
# # 🅱️  APPROCHE B — Split par population (Unseen Users)
# # ============================================================================

# def get_dataloaders_approach_B(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
#     random_seed: int   = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """
#     Approche B — Split par population (utilisateurs inconnus)
#     ══════════════════════════════════════════════════════════
#     80 % des utilisateurs complets → TRAIN.
#     20 % des utilisateurs complets → TEST (jamais vus).

#     Normalisation : StandardScaler fitté sur train uniquement.
#     """
#     dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
#     unique_users = np.unique(dataset.user_ids)

#     rng            = np.random.default_rng(random_seed)
#     shuffled_users = rng.permutation(unique_users)
#     split_pt       = int(len(shuffled_users) * train_ratio)
#     train_users    = set(shuffled_users[:split_pt])
#     test_users     = set(shuffled_users[split_pt:])

#     train_indices = [i for i, u in enumerate(dataset.user_ids) if u in train_users]
#     test_indices  = [i for i, u in enumerate(dataset.user_ids) if u in test_users]

#     _print_split_report(
#         dataset, train_indices, test_indices,
#         "Approche B — Split par Population (Unseen Users)",
#         extra_info=(
#             f"Users Train : {len(train_users):,}"
#             f"  │  Users Test : {len(test_users):,}"
#             f"  │  Seed : {random_seed}"
#         ),
#     )

#     # ── Normalisation fittée sur train uniquement ─────────────────────────────
#     dataset.fit_and_apply_scaler(train_indices)

#     return _build_loaders(dataset, train_indices, test_indices, batch_size)


# # ============================================================================
# # 🔄  ALIAS FL — compatibilité client.py Flower
# # ============================================================================

# def get_split_dataloaders(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
#     random_seed: int   = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """Alias pour client.py Flower → délègue vers Approche B."""
#     return get_dataloaders_approach_B(
#         data_dir    = data_dir,
#         train_ratio = train_ratio,
#         batch_size  = batch_size,
#         seq_len     = seq_len,
#         random_seed = random_seed,
#     )


# # ============================================================================
# # 🧪 ZONE DE TEST
# # ============================================================================

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Test du Dataloader de fraude")
#     parser.add_argument("--data_dir",   type=str, default="data/node_1/tensors")
#     parser.add_argument("--approach",   type=str, default="A", choices=["A", "B"])
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--seq_len",    type=int, default=5)
#     args = parser.parse_args()

#     print(f"\n🧪 Test : Approche {args.approach}")

#     fn = get_dataloaders_approach_A if args.approach == "A" else get_dataloaders_approach_B
#     train_loader, test_loader = fn(
#         data_dir   = args.data_dir,
#         batch_size = args.batch_size,
#         seq_len    = args.seq_len,
#     )

#     print(f"  Batches → Train : {len(train_loader)} | Test : {len(test_loader)}")
#     x_sample, y_sample = next(iter(train_loader))
#     print(f"  Forme X : {x_sample.shape}  → attendu ({args.batch_size}, {args.seq_len}, 26)")
#     print(f"  Forme y : {y_sample.shape}  → attendu ({args.batch_size}, 1)")
#     print(f"  X min={x_sample.min():.3f}  max={x_sample.max():.3f}  ← doit être ~[-3, 3]")
#     print(f"  Fraudes dans le batch : {int(y_sample.sum())}/{len(y_sample)}")
#     print(f"\n✅ Dataloader opérationnel !")



# """
# dataloader.py — Pipeline de chargement de données pour la détection de fraude (FL)
# ====================================================================================
# Stratégies de split disponibles :
#   - Approche A : Split temporel INTRA-utilisateur  (70% train / 10% val / 20% test)
#   - Approche B : Split par POPULATION              (80% users connus / 20% users inconnus)

# Contrainte commune :
#   - Séquences de longueur SEQ_LEN=5, jamais de mélange inter-utilisateurs.
#   - Colonne 0 des .npy = PAN_ID (retirée avant le modèle GRU).

# Normalisation :
#   - RobustScaler fitté UNIQUEMENT sur les indices train + clip [-5, 5].
#   - Résout le NaN causé par Time_Delta_Sec (~31M) et Velocity_Sum_24H (~2.7M).

# Split 70/10/20 (Approche A) :
#   - Train      : early stopping + apprentissage
#   - Validation : checkpoint + early stopping (jamais utilisé pour apprendre)
#   - Test       : évaluation finale UNIQUEMENT (résultat honnête, non biaisé)
# """

# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# from pathlib import Path


# # ============================================================================
# # 📦 DATASET
# # ============================================================================

# class FraudSequenceDataset(Dataset):
#     def __init__(self, data_dir: str | Path, sequence_length: int = 5):
#         self.data_dir        = Path(data_dir)
#         self.sequence_length = sequence_length

#         x_files = sorted(self.data_dir.glob("X_batch_*.npy"))
#         y_files = sorted(self.data_dir.glob("y_batch_*.npy"))

#         if not x_files:
#             raise FileNotFoundError(f"❌ Aucun fichier X_batch_*.npy trouvé dans {self.data_dir}")
#         if len(x_files) != len(y_files):
#             raise ValueError(f"❌ Fichiers X ({len(x_files)}) ≠ Y ({len(y_files)})")

#         all_x, all_y = [], []
#         for xf, yf in zip(x_files, y_files):
#             all_x.append(np.load(xf))
#             all_y.append(np.load(yf))

#         raw_X = np.vstack(all_x).astype(np.float32)
#         raw_y = np.vstack(all_y).astype(np.float32).flatten()

#         # Tri stable par PAN_ID → ordre chronologique intra-user préservé
#         sort_idx      = np.argsort(raw_X[:, 0], kind="stable")
#         raw_X         = raw_X[sort_idx]
#         raw_y         = raw_y[sort_idx]

#         self.user_ids = raw_X[:, 0]
#         self.features = raw_X[:, 1:]   # (N, 26) — pas encore normalisé
#         self.y        = raw_y

#         n_users = len(np.unique(self.user_ids))
#         print(
#             f"📦 Dataset chargé : {len(self.features):,} transactions"
#             f" | {n_users:,} utilisateurs uniques"
#             f" | {int(self.y.sum()):,} fraudes ({100*self.y.mean():.2f} %)"
#         )

#     def fit_and_apply_scaler(self, train_indices: list) -> None:
#         """
#         RobustScaler fitté sur train uniquement + clip [-5, 5].
#         Évite le data leakage et les NaN FP16.
#         """
#         train_features = self.features[train_indices]

#         median = np.median(train_features, axis=0)
#         q75, q25 = np.percentile(train_features, [75, 25], axis=0)
#         iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)

#         self.features = (self.features - median) / iqr
#         self.features = np.clip(self.features, -5.0, 5.0)   # coupe les outliers

#         print(f"📐 Normalisation appliquée (fitté sur {len(train_indices):,} txns train)")
#         print(f"   Avant : min={train_features.min():.1f}  max={train_features.max():.1f}")
#         print(f"   Après : min={self.features.min():.3f}  max={self.features.max():.3f}")

#     def __len__(self) -> int:
#         return len(self.features)

#     def __getitem__(self, idx: int):
#         current_user = self.user_ids[idx]
#         sequence     = []

#         for lag in range(self.sequence_length - 1, -1, -1):
#             target_idx = idx - lag
#             if target_idx < 0 or self.user_ids[target_idx] != current_user:
#                 sequence.append(np.zeros(self.features.shape[1], dtype=np.float32))
#             else:
#                 sequence.append(self.features[target_idx])

#         x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
#         y_tensor = torch.tensor([self.y[idx]],       dtype=torch.float32)
#         return x_tensor, y_tensor


# # ============================================================================
# # 🔧 UTILITAIRES
# # ============================================================================

# def _make_loader(dataset, indices, batch_size, shuffle) -> DataLoader:
#     return DataLoader(
#         Subset(dataset, indices),
#         batch_size  = batch_size,
#         shuffle     = shuffle,
#         num_workers = 0,
#         pin_memory  = True,
#         drop_last   = False,
#     )

# def _print_split_report(dataset, splits: dict, approach_name: str, extra_info: str = ""):
#     y = dataset.y
#     print(f"\n{'─'*58}")
#     print(f"  📊  {approach_name}")
#     if extra_info:
#         print(f"  {extra_info}")
#     print(f"{'─'*58}")
#     for name, indices in splits.items():
#         n      = len(indices)
#         fraud  = float(sum(y[i] for i in indices))
#         pct    = 100 * fraud / max(n, 1)
#         print(f"  {name:12s} │ {n:>7,} transactions │ {int(fraud):>5} fraudes ({pct:.2f} %)")
#     print(f"{'─'*58}\n")


# # ============================================================================
# # 🅰️  APPROCHE A — Split temporel INTRA-utilisateur  70 / 10 / 20
# # ============================================================================

# def get_dataloaders_approach_A(
#     data_dir:    str | Path,
#     train_ratio: float = 0.70,
#     val_ratio:   float = 0.10,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Approche A — Split temporel 70 / 10 / 20 par utilisateur
#     ══════════════════════════════════════════════════════════
#     Pour chaque utilisateur (ordre chronologique) :
#       - 70% premières transactions → TRAIN
#       - 10% suivantes              → VALIDATION  (early stopping + checkpoint)
#       - 20% dernières              → TEST        (évaluation finale honnête)

#     Retourne
#     ────────
#     (train_loader, val_loader, test_loader)
#     """
#     dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
#     user_ids_arr = dataset.user_ids

#     # Split vectorisé O(N) — pas de boucle np.where
#     change_mask  = np.concatenate(([True], user_ids_arr[1:] != user_ids_arr[:-1]))
#     group_starts = np.where(change_mask)[0]
#     group_ends   = np.concatenate((group_starts[1:], [len(user_ids_arr)]))
#     group_sizes  = group_ends - group_starts

#     train_end_pts = np.maximum(1,               (group_sizes * train_ratio).astype(int))
#     val_end_pts   = np.maximum(train_end_pts+1, (group_sizes * (train_ratio + val_ratio)).astype(int))

#     train_indices, val_indices, test_indices = [], [], []

#     for start, size, t_end, v_end in zip(group_starts, group_sizes, train_end_pts, val_end_pts):
#         idx = np.arange(start, start + size)
#         train_indices.append(idx[:t_end])
#         if v_end > t_end:
#             val_indices.append(idx[t_end:v_end])
#         if v_end < size:
#             test_indices.append(idx[v_end:])

#     train_indices = np.concatenate(train_indices).tolist()
#     val_indices   = np.concatenate(val_indices).tolist()   if val_indices   else []
#     test_indices  = np.concatenate(test_indices).tolist()  if test_indices  else []

#     _print_split_report(
#         dataset,
#         {"Train (70%)": train_indices, "Val   (10%)": val_indices, "Test  (20%)": test_indices},
#         "Approche A — Split Temporel Intra-User  70/10/20",
#         extra_info=f"Users : {len(group_starts):,}",
#     )

#     # Normalisation fittée sur train uniquement
#     dataset.fit_and_apply_scaler(train_indices)

#     return (
#         _make_loader(dataset, train_indices, batch_size, shuffle=True),
#         _make_loader(dataset, val_indices,   batch_size, shuffle=False),
#         _make_loader(dataset, test_indices,  batch_size, shuffle=False),
#     )


# # ============================================================================
# # 🅱️  APPROCHE B — Split par population (Unseen Users)  80 / 20
# # ============================================================================

# def get_dataloaders_approach_B(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
#     random_seed: int   = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """
#     Approche B — Split par population
#     ════════════════════════════════════
#     80 % des utilisateurs complets → TRAIN
#     20 % des utilisateurs complets → TEST (jamais vus)

#     Retourne
#     ────────
#     (train_loader, test_loader)
#     """
#     dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
#     unique_users = np.unique(dataset.user_ids)

#     rng            = np.random.default_rng(random_seed)
#     shuffled_users = rng.permutation(unique_users)
#     split_pt       = int(len(shuffled_users) * train_ratio)
#     train_users    = set(shuffled_users[:split_pt])
#     test_users     = set(shuffled_users[split_pt:])

#     train_indices = [i for i, u in enumerate(dataset.user_ids) if u in train_users]
#     test_indices  = [i for i, u in enumerate(dataset.user_ids) if u in test_users]

#     _print_split_report(
#         dataset,
#         {"Train (80%)": train_indices, "Test  (20%)": test_indices},
#         "Approche B — Split par Population (Unseen Users)",
#         extra_info=f"Users Train : {len(train_users):,}  │  Users Test : {len(test_users):,}  │  Seed : {random_seed}",
#     )

#     dataset.fit_and_apply_scaler(train_indices)

#     return (
#         _make_loader(dataset, train_indices, batch_size, shuffle=True),
#         _make_loader(dataset, test_indices,  batch_size, shuffle=False),
#     )


# # ============================================================================
# # 🔄  ALIAS FL — client.py Flower
# # ============================================================================

# def get_split_dataloaders(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
#     random_seed: int   = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """Alias pour client.py Flower → délègue vers Approche B."""
#     return get_dataloaders_approach_B(
#         data_dir=data_dir, train_ratio=train_ratio,
#         batch_size=batch_size, seq_len=seq_len, random_seed=random_seed,
#     )


# # ============================================================================
# # 🧪 ZONE DE TEST
# # ============================================================================

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir",   default="data/node_1/tensors")
#     parser.add_argument("--approach",   default="A", choices=["A", "B"])
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--seq_len",    type=int, default=5)
#     args = parser.parse_args()

#     if args.approach == "A":
#         train_l, val_l, test_l = get_dataloaders_approach_A(args.data_dir, batch_size=args.batch_size)
#         print(f"  Train: {len(train_l)} batches | Val: {len(val_l)} batches | Test: {len(test_l)} batches")
#         x, y = next(iter(val_l))
#         print(f"  Val X : {x.shape}  |  X range : [{x.min():.2f}, {x.max():.2f}]")
#     else:
#         train_l, test_l = get_dataloaders_approach_B(args.data_dir, batch_size=args.batch_size)
#         print(f"  Train: {len(train_l)} batches | Test: {len(test_l)} batches")




# """
# dataloader.py — Pipeline de chargement de données pour la détection de fraude (FL)
# ====================================================================================
# Stratégies de split disponibles :
#   - Approche A : Split temporel INTRA-utilisateur  (70% train / 10% val / 20% test)
#   - Approche B : Split par POPULATION              (80% users connus / 20% users inconnus)

# Contrainte commune :
#   - Séquences de longueur SEQ_LEN=5, jamais de mélange inter-utilisateurs.
#   - Colonne 0 des .npy = PAN_ID (retirée avant le modèle GRU).

# Normalisation :
#   - RobustScaler fitté UNIQUEMENT sur les indices train + clip [-5, 5].
#   - Résout le NaN causé par Time_Delta_Sec (~31M) et Velocity_Sum_24H (~2.7M).

# Split 70/10/20 (Approche A) :
#   - Train      : early stopping + apprentissage
#   - Validation : checkpoint + early stopping (jamais utilisé pour apprendre)
#   - Test       : évaluation finale UNIQUEMENT (résultat honnête, non biaisé)
# """

# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# from pathlib import Path


# # ============================================================================
# # 📦 DATASET
# # ============================================================================

# class FraudSequenceDataset(Dataset):
#     def __init__(self, data_dir: str | Path, sequence_length: int = 5):
#         self.data_dir        = Path(data_dir)
#         self.sequence_length = sequence_length

#         x_files = sorted(self.data_dir.glob("X_batch_*.npy"))
#         y_files = sorted(self.data_dir.glob("y_batch_*.npy"))

#         if not x_files:
#             raise FileNotFoundError(f"❌ Aucun fichier X_batch_*.npy trouvé dans {self.data_dir}")
#         if len(x_files) != len(y_files):
#             raise ValueError(f"❌ Fichiers X ({len(x_files)}) ≠ Y ({len(y_files)})")

#         all_x, all_y = [], []
#         for xf, yf in zip(x_files, y_files):
#             all_x.append(np.load(xf))
#             all_y.append(np.load(yf))

#         raw_X = np.vstack(all_x).astype(np.float32)
#         raw_y = np.vstack(all_y).astype(np.float32).flatten()

#         # Tri stable par PAN_ID → ordre chronologique intra-user préservé
#         sort_idx      = np.argsort(raw_X[:, 0], kind="stable")
#         raw_X         = raw_X[sort_idx]
#         raw_y         = raw_y[sort_idx]

#         self.user_ids = raw_X[:, 0]
#         self.features = raw_X[:, 1:]   # (N, 26) — pas encore normalisé
#         self.y        = raw_y

#         n_users = len(np.unique(self.user_ids))
#         print(
#             f"📦 Dataset chargé : {len(self.features):,} transactions"
#             f" | {n_users:,} utilisateurs uniques"
#             f" | {int(self.y.sum()):,} fraudes ({100*self.y.mean():.2f} %)"
#         )

#     def fit_and_apply_scaler(self, train_indices: list) -> None:
#         """
#         RobustScaler fitté sur train uniquement + clip [-5, 5].
#         Évite le data leakage et les NaN FP16.
#         """
#         train_features = self.features[train_indices]

#         median = np.median(train_features, axis=0)
#         q75, q25 = np.percentile(train_features, [75, 25], axis=0)
#         iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)

#         self.features = (self.features - median) / iqr
#         self.features = np.clip(self.features, -5.0, 5.0)   # coupe les outliers

#         print(f"📐 Normalisation appliquée (fitté sur {len(train_indices):,} txns train)")
#         print(f"   Avant : min={train_features.min():.1f}  max={train_features.max():.1f}")
#         print(f"   Après : min={self.features.min():.3f}  max={self.features.max():.3f}")

#     def __len__(self) -> int:
#         return len(self.features)

#     def __getitem__(self, idx: int):
#         current_user = self.user_ids[idx]
#         sequence     = []

#         for lag in range(self.sequence_length - 1, -1, -1):
#             target_idx = idx - lag
#             if target_idx < 0 or self.user_ids[target_idx] != current_user:
#                 sequence.append(np.zeros(self.features.shape[1], dtype=np.float32))
#             else:
#                 sequence.append(self.features[target_idx])

#         x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
#         y_tensor = torch.tensor([self.y[idx]],       dtype=torch.float32)
#         return x_tensor, y_tensor


# # ============================================================================
# # 🔧 UTILITAIRES
# # ============================================================================

# def _make_loader(dataset, indices, batch_size, shuffle) -> DataLoader:
#     return DataLoader(
#         Subset(dataset, indices),
#         batch_size  = batch_size,
#         shuffle     = shuffle,
#         num_workers = 0,
#         pin_memory  = True,
#         drop_last   = False,
#     )

# def _print_split_report(dataset, splits: dict, approach_name: str, extra_info: str = ""):
#     y = dataset.y
#     print(f"\n{'─'*58}")
#     print(f"  📊  {approach_name}")
#     if extra_info:
#         print(f"  {extra_info}")
#     print(f"{'─'*58}")
#     for name, indices in splits.items():
#         n      = len(indices)
#         fraud  = float(sum(y[i] for i in indices))
#         pct    = 100 * fraud / max(n, 1)
#         print(f"  {name:12s} │ {n:>7,} transactions │ {int(fraud):>5} fraudes ({pct:.2f} %)")
#     print(f"{'─'*58}\n")


# # ============================================================================
# # 🅰️  APPROCHE A — Split temporel INTRA-utilisateur  70 / 10 / 20
# # ============================================================================

# def get_dataloaders_approach_A(
#     data_dir:    str | Path,
#     train_ratio: float = 0.70,
#     val_ratio:   float = 0.10,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Approche A — Split temporel 70 / 10 / 20 par utilisateur
#     ══════════════════════════════════════════════════════════
#     Pour chaque utilisateur (ordre chronologique) :
#       - 70% premières transactions → TRAIN
#       - 10% suivantes              → VALIDATION  (early stopping + checkpoint)
#       - 20% dernières              → TEST        (évaluation finale honnête)

#     Retourne
#     ────────
#     (train_loader, val_loader, test_loader)
#     """
#     dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
#     user_ids_arr = dataset.user_ids

#     # Split vectorisé O(N) — pas de boucle np.where
#     change_mask  = np.concatenate(([True], user_ids_arr[1:] != user_ids_arr[:-1]))
#     group_starts = np.where(change_mask)[0]
#     group_ends   = np.concatenate((group_starts[1:], [len(user_ids_arr)]))
#     group_sizes  = group_ends - group_starts

#     train_end_pts = np.maximum(1,               (group_sizes * train_ratio).astype(int))
#     val_end_pts   = np.maximum(train_end_pts+1, (group_sizes * (train_ratio + val_ratio)).astype(int))

#     train_indices, val_indices, test_indices = [], [], []

#     for start, size, t_end, v_end in zip(group_starts, group_sizes, train_end_pts, val_end_pts):
#         idx = np.arange(start, start + size)
#         train_indices.append(idx[:t_end])
#         if v_end > t_end:
#             val_indices.append(idx[t_end:v_end])
#         if v_end < size:
#             test_indices.append(idx[v_end:])

#     train_indices = np.concatenate(train_indices).tolist()
#     val_indices   = np.concatenate(val_indices).tolist()   if val_indices   else []
#     test_indices  = np.concatenate(test_indices).tolist()  if test_indices  else []

#     _print_split_report(
#         dataset,
#         {"Train (70%)": train_indices, "Val   (10%)": val_indices, "Test  (20%)": test_indices},
#         "Approche A — Split Temporel Intra-User  70/10/20",
#         extra_info=f"Users : {len(group_starts):,}",
#     )

#     # Normalisation fittée sur train uniquement
#     dataset.fit_and_apply_scaler(train_indices)

#     return (
#         _make_loader(dataset, train_indices, batch_size, shuffle=True),
#         _make_loader(dataset, val_indices,   batch_size, shuffle=False),
#         _make_loader(dataset, test_indices,  batch_size, shuffle=False),
#     )


# # ============================================================================
# # 🅱️  APPROCHE B — Split stratifié par taille ET taux de fraude
# # ============================================================================

# def _stratified_user_split(
#     dataset:     FraudSequenceDataset,
#     test_ratio:  float = 0.20,
#     n_size_bins: int   = 2,
#     n_fraud_bins:int   = 2,
# ) -> tuple[set, set]:
#     """
#     Divise les utilisateurs en train/test de façon stratifiée.

#     Chaque utilisateur est classé dans un stratum selon :
#       - sa taille  (nb de transactions : small / large)
#       - son taux de fraude (low / high)

#     On prend exactement test_ratio % de chaque stratum pour le test.
#     Garantit que train et test ont la même distribution de tailles
#     ET le même taux de fraude — sans dépendre d'un seed aléatoire.
#     """
#     unique_users = np.unique(dataset.user_ids)

#     # ── Calcul des stats par user ────────────────────────────────────────────
#     user_stats = []
#     for user in unique_users:
#         mask    = dataset.user_ids == user
#         n_txn   = mask.sum()
#         n_fraud = dataset.y[mask].sum()
#         user_stats.append({
#             "user"       : user,
#             "n_txn"      : int(n_txn),
#             "fraud_rate" : float(n_fraud / n_txn) if n_txn > 0 else 0.0,
#         })

#     n_txns      = np.array([s["n_txn"]      for s in user_stats])
#     fraud_rates = np.array([s["fraud_rate"] for s in user_stats])

#     # ── Création des bins ────────────────────────────────────────────────────
#     # np.array_split sur les indices triés → bins de taille égale
#     size_sorted  = np.argsort(n_txns)
#     fraud_sorted = np.argsort(fraud_rates)

#     size_bins  = np.array_split(size_sorted,  n_size_bins)
#     fraud_bins = np.array_split(fraud_sorted, n_fraud_bins)

#     # Assignation des bins à chaque user
#     size_label  = np.empty(len(unique_users), dtype=int)
#     fraud_label = np.empty(len(unique_users), dtype=int)
#     for bin_idx, indices in enumerate(size_bins):
#         size_label[indices]  = bin_idx
#     for bin_idx, indices in enumerate(fraud_bins):
#         fraud_label[indices] = bin_idx

#     # ── Split test_ratio % dans chaque stratum ───────────────────────────────
#     train_users : set = set()
#     test_users  : set = set()

#     strata_report = []
#     for s_bin in range(n_size_bins):
#         for f_bin in range(n_fraud_bins):
#             mask_stratum = (size_label == s_bin) & (fraud_label == f_bin)
#             stratum_idx  = np.where(mask_stratum)[0]

#             if len(stratum_idx) == 0:
#                 continue

#             # Tri par taille dans le stratum puis 1 sur N pour le test
#             # → déterministe, pas de random
#             stratum_sorted = stratum_idx[np.argsort(n_txns[stratum_idx])]
#             step           = max(1, round(1.0 / test_ratio))
#             test_mask      = np.zeros(len(stratum_sorted), dtype=bool)
#             test_mask[::step] = True

#             for i, idx in enumerate(stratum_sorted):
#                 user = user_stats[idx]["user"]
#                 if test_mask[i]:
#                     test_users.add(user)
#                 else:
#                     train_users.add(user)

#             strata_report.append({
#                 "stratum": f"size={s_bin} fraud={f_bin}",
#                 "total"  : len(stratum_idx),
#                 "test"   : int(test_mask.sum()),
#             })

#     # ── Log des stratums ─────────────────────────────────────────────────────
#     size_labels  = ["small", "large"]
#     fraud_labels = ["low fraud", "high fraud"]
#     print(f"  Stratification des {len(unique_users)} utilisateurs :")
#     for s_bin in range(n_size_bins):
#         for f_bin in range(n_fraud_bins):
#             mask_stratum = (size_label == s_bin) & (fraud_label == f_bin)
#             n_total = mask_stratum.sum()
#             n_test  = sum(1 for u in test_users
#                           if user_stats[list(unique_users).index(u)]["user"] == u
#                           and size_label[list(unique_users).index(u)] == s_bin
#                           and fraud_label[list(unique_users).index(u)] == f_bin)
#             label = f"{size_labels[s_bin]}, {fraud_labels[f_bin]}"
#             print(f"    {label:25s} : {n_total:3d} users → {n_total - n_test:3d} train / {n_test:2d} test")

#     return train_users, test_users


# def get_dataloaders_approach_B(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
# ) -> tuple[DataLoader, DataLoader]:
#     """
#     Approche B — Split stratifié par taille ET taux de fraude
#     ══════════════════════════════════════════════════════════
#     Les utilisateurs sont répartis en 4 groupes (stratums) :
#       - small + low fraud   → petit client, peu fraudé
#       - small + high fraud  → petit client, très fraudé
#       - large + low fraud   → gros client, peu fraudé
#       - large + high fraud  → gros client, très fraudé

#     Dans chaque stratum : ~20% des users → TEST, ~80% → TRAIN.
#     Split déterministe (pas de seed aléatoire).

#     Retourne
#     ────────
#     (train_loader, test_loader)
#     """
#     dataset = FraudSequenceDataset(data_dir, sequence_length=seq_len)

#     train_users, test_users = _stratified_user_split(
#         dataset,
#         test_ratio   = 1.0 - train_ratio,
#         n_size_bins  = 2,
#         n_fraud_bins = 2,
#     )

#     train_indices = [i for i, u in enumerate(dataset.user_ids) if u in train_users]
#     test_indices  = [i for i, u in enumerate(dataset.user_ids) if u in test_users]

#     _print_split_report(
#         dataset,
#         {"Train (80%)": train_indices, "Test  (20%)": test_indices},
#         "Approche B — Split Stratifié (taille + taux fraude)",
#         extra_info=f"Users Train : {len(train_users):,}  │  Users Test : {len(test_users):,}  │  Déterministe",
#     )

#     dataset.fit_and_apply_scaler(train_indices)

#     return (
#         _make_loader(dataset, train_indices, batch_size, shuffle=True),
#         _make_loader(dataset, test_indices,  batch_size, shuffle=False),
#     )


# # ============================================================================
# # 🔄  ALIAS FL — client.py Flower
# # ============================================================================

# def get_split_dataloaders(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
#     random_seed: int   = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """Alias pour client.py Flower → délègue vers Approche B."""
#     return get_dataloaders_approach_B(
#         data_dir=data_dir, train_ratio=train_ratio,
#         batch_size=batch_size, seq_len=seq_len, random_seed=random_seed,
#     )


# # ============================================================================
# # 🧪 ZONE DE TEST
# # ============================================================================

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir",   default="data/node_1/tensors")
#     parser.add_argument("--approach",   default="A", choices=["A", "B"])
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--seq_len",    type=int, default=5)
#     args = parser.parse_args()

#     if args.approach == "A":
#         train_l, val_l, test_l = get_dataloaders_approach_A(args.data_dir, batch_size=args.batch_size)
#         print(f"  Train: {len(train_l)} batches | Val: {len(val_l)} batches | Test: {len(test_l)} batches")
#         x, y = next(iter(val_l))
#         print(f"  Val X : {x.shape}  |  X range : [{x.min():.2f}, {x.max():.2f}]")
#     else:
#         train_l, test_l = get_dataloaders_approach_B(args.data_dir, batch_size=args.batch_size)
#         print(f"  Train: {len(train_l)} batches | Test: {len(test_l)} batches")




# """
# dataloader.py — Pipeline de chargement de données pour la détection de fraude (FL)
# ====================================================================================
# Stratégies de split disponibles :
#   - Approche A : Split temporel INTRA-utilisateur  (70% train / 10% val / 20% test)
#   - Approche B : Split par POPULATION              (80% users connus / 20% users inconnus)

# Contrainte commune :
#   - Séquences de longueur SEQ_LEN=5, jamais de mélange inter-utilisateurs.
#   - Colonne 0 des .npy = PAN_ID (retirée avant le modèle GRU).

# Normalisation :
#   - RobustScaler fitté UNIQUEMENT sur les indices train + clip [-5, 5].
#   - Résout le NaN causé par Time_Delta_Sec (~31M) et Velocity_Sum_24H (~2.7M).

# Split 70/10/20 (Approche A) :
#   - Train      : early stopping + apprentissage
#   - Validation : checkpoint + early stopping (jamais utilisé pour apprendre)
#   - Test       : évaluation finale UNIQUEMENT (résultat honnête, non biaisé)
# """

# import torch
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# from pathlib import Path


# # ============================================================================
# # 📦 DATASET
# # ============================================================================

# class FraudSequenceDataset(Dataset):
#     def __init__(self, data_dir: str | Path, sequence_length: int = 5):
#         self.data_dir        = Path(data_dir)
#         self.sequence_length = sequence_length

#         x_files = sorted(self.data_dir.glob("X_batch_*.npy"))
#         y_files = sorted(self.data_dir.glob("y_batch_*.npy"))

#         if not x_files:
#             raise FileNotFoundError(f"❌ Aucun fichier X_batch_*.npy trouvé dans {self.data_dir}")
#         if len(x_files) != len(y_files):
#             raise ValueError(f"❌ Fichiers X ({len(x_files)}) ≠ Y ({len(y_files)})")

#         all_x, all_y = [], []
#         for xf, yf in zip(x_files, y_files):
#             all_x.append(np.load(xf))
#             all_y.append(np.load(yf))

#         raw_X = np.vstack(all_x).astype(np.float32)
#         raw_y = np.vstack(all_y).astype(np.float32).flatten()

#         # Tri stable par PAN_ID → ordre chronologique intra-user préservé
#         sort_idx      = np.argsort(raw_X[:, 0], kind="stable")
#         raw_X         = raw_X[sort_idx]
#         raw_y         = raw_y[sort_idx]

#         self.user_ids = raw_X[:, 0]
#         self.features = raw_X[:, 1:]   # (N, 26) — pas encore normalisé
#         self.y        = raw_y

#         n_users = len(np.unique(self.user_ids))
#         print(
#             f"📦 Dataset chargé : {len(self.features):,} transactions"
#             f" | {n_users:,} utilisateurs uniques"
#             f" | {int(self.y.sum()):,} fraudes ({100*self.y.mean():.2f} %)"
#         )

#     def fit_and_apply_scaler(self, train_indices: list) -> None:
#         """
#         RobustScaler fitté sur train uniquement + clip [-5, 5].
#         Évite le data leakage et les NaN FP16.
#         """
#         train_features = self.features[train_indices]

#         median = np.median(train_features, axis=0)
#         q75, q25 = np.percentile(train_features, [75, 25], axis=0)
#         iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)

#         self.features = (self.features - median) / iqr
#         self.features = np.clip(self.features, -5.0, 5.0)   # coupe les outliers

#         print(f"📐 Normalisation appliquée (fitté sur {len(train_indices):,} txns train)")
#         print(f"   Avant : min={train_features.min():.1f}  max={train_features.max():.1f}")
#         print(f"   Après : min={self.features.min():.3f}  max={self.features.max():.3f}")

#     def __len__(self) -> int:
#         return len(self.features)

#     def __getitem__(self, idx: int):
#         current_user = self.user_ids[idx]
#         sequence     = []

#         for lag in range(self.sequence_length - 1, -1, -1):
#             target_idx = idx - lag
#             if target_idx < 0 or self.user_ids[target_idx] != current_user:
#                 sequence.append(np.zeros(self.features.shape[1], dtype=np.float32))
#             else:
#                 sequence.append(self.features[target_idx])

#         x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
#         y_tensor = torch.tensor([self.y[idx]],       dtype=torch.float32)
#         return x_tensor, y_tensor


# # ============================================================================
# # 🔧 UTILITAIRES
# # ============================================================================

# def _make_loader(dataset, indices, batch_size, shuffle) -> DataLoader:
#     return DataLoader(
#         Subset(dataset, indices),
#         batch_size  = batch_size,
#         shuffle     = shuffle,
#         num_workers = 0,
#         pin_memory  = True,
#         drop_last   = False,
#     )

# def _print_split_report(dataset, splits: dict, approach_name: str, extra_info: str = ""):
#     y = dataset.y
#     print(f"\n{'─'*58}")
#     print(f"  📊  {approach_name}")
#     if extra_info:
#         print(f"  {extra_info}")
#     print(f"{'─'*58}")
#     for name, indices in splits.items():
#         n      = len(indices)
#         fraud  = float(sum(y[i] for i in indices))
#         pct    = 100 * fraud / max(n, 1)
#         print(f"  {name:12s} │ {n:>7,} transactions │ {int(fraud):>5} fraudes ({pct:.2f} %)")
#     print(f"{'─'*58}\n")


# # ============================================================================
# # 🅰️  APPROCHE A — Split temporel INTRA-utilisateur  70 / 10 / 20
# # ============================================================================

# def get_dataloaders_approach_A(
#     data_dir:    str | Path,
#     train_ratio: float = 0.70,
#     val_ratio:   float = 0.10,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Approche A — Split temporel 70 / 10 / 20 par utilisateur
#     ══════════════════════════════════════════════════════════
#     Pour chaque utilisateur (ordre chronologique) :
#       - 70% premières transactions → TRAIN
#       - 10% suivantes              → VALIDATION  (early stopping + checkpoint)
#       - 20% dernières              → TEST        (évaluation finale honnête)

#     Retourne
#     ────────
#     (train_loader, val_loader, test_loader)
#     """
#     dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
#     user_ids_arr = dataset.user_ids

#     # Split vectorisé O(N) — pas de boucle np.where
#     change_mask  = np.concatenate(([True], user_ids_arr[1:] != user_ids_arr[:-1]))
#     group_starts = np.where(change_mask)[0]
#     group_ends   = np.concatenate((group_starts[1:], [len(user_ids_arr)]))
#     group_sizes  = group_ends - group_starts

#     train_end_pts = np.maximum(1,               (group_sizes * train_ratio).astype(int))
#     val_end_pts   = np.maximum(train_end_pts+1, (group_sizes * (train_ratio + val_ratio)).astype(int))

#     train_indices, val_indices, test_indices = [], [], []

#     for start, size, t_end, v_end in zip(group_starts, group_sizes, train_end_pts, val_end_pts):
#         idx = np.arange(start, start + size)
#         train_indices.append(idx[:t_end])
#         if v_end > t_end:
#             val_indices.append(idx[t_end:v_end])
#         if v_end < size:
#             test_indices.append(idx[v_end:])

#     train_indices = np.concatenate(train_indices).tolist()
#     val_indices   = np.concatenate(val_indices).tolist()   if val_indices   else []
#     test_indices  = np.concatenate(test_indices).tolist()  if test_indices  else []

#     _print_split_report(
#         dataset,
#         {"Train (70%)": train_indices, "Val   (10%)": val_indices, "Test  (20%)": test_indices},
#         "Approche A — Split Temporel Intra-User  70/10/20",
#         extra_info=f"Users : {len(group_starts):,}",
#     )

#     # Normalisation fittée sur train uniquement
#     dataset.fit_and_apply_scaler(train_indices)

#     return (
#         _make_loader(dataset, train_indices, batch_size, shuffle=True),
#         _make_loader(dataset, val_indices,   batch_size, shuffle=False),
#         _make_loader(dataset, test_indices,  batch_size, shuffle=False),
#     )


# # ============================================================================
# # 🅱️  APPROCHE B — Split stratifié par taille ET taux de fraude
# # ============================================================================

# def _stratified_user_split(
#     dataset:     FraudSequenceDataset,
#     test_ratio:  float = 0.20,
#     n_size_bins: int   = 2,
#     n_fraud_bins:int   = 2,
# ) -> tuple[set, set]:
#     """
#     Divise les utilisateurs en train/test de façon stratifiée.

#     Chaque utilisateur est classé dans un stratum selon :
#       - sa taille  (nb de transactions : small / large)
#       - son taux de fraude (low / high)

#     On prend exactement test_ratio % de chaque stratum pour le test.
#     Garantit que train et test ont la même distribution de tailles
#     ET le même taux de fraude — sans dépendre d'un seed aléatoire.
#     """
#     unique_users = np.unique(dataset.user_ids)

#     # ── Calcul des stats par user ────────────────────────────────────────────
#     user_stats = []
#     for user in unique_users:
#         mask    = dataset.user_ids == user
#         n_txn   = mask.sum()
#         n_fraud = dataset.y[mask].sum()
#         user_stats.append({
#             "user"       : user,
#             "n_txn"      : int(n_txn),
#             "fraud_rate" : float(n_fraud / n_txn) if n_txn > 0 else 0.0,
#         })

#     n_txns      = np.array([s["n_txn"]      for s in user_stats])
#     fraud_rates = np.array([s["fraud_rate"] for s in user_stats])

#     # ── Création des bins ────────────────────────────────────────────────────
#     # np.array_split sur les indices triés → bins de taille égale
#     size_sorted  = np.argsort(n_txns)
#     fraud_sorted = np.argsort(fraud_rates)

#     size_bins  = np.array_split(size_sorted,  n_size_bins)
#     fraud_bins = np.array_split(fraud_sorted, n_fraud_bins)

#     # Assignation des bins à chaque user
#     size_label  = np.empty(len(unique_users), dtype=int)
#     fraud_label = np.empty(len(unique_users), dtype=int)
#     for bin_idx, indices in enumerate(size_bins):
#         size_label[indices]  = bin_idx
#     for bin_idx, indices in enumerate(fraud_bins):
#         fraud_label[indices] = bin_idx

#     # ── Split test_ratio % dans chaque stratum ───────────────────────────────
#     train_users : set = set()
#     test_users  : set = set()

#     strata_report = []
#     for s_bin in range(n_size_bins):
#         for f_bin in range(n_fraud_bins):
#             mask_stratum = (size_label == s_bin) & (fraud_label == f_bin)
#             stratum_idx  = np.where(mask_stratum)[0]

#             if len(stratum_idx) == 0:
#                 continue

#             # Tri par taille dans le stratum puis 1 sur N pour le test
#             # → déterministe, pas de random
#             stratum_sorted = stratum_idx[np.argsort(n_txns[stratum_idx])]
#             step           = max(1, round(1.0 / test_ratio))
#             test_mask      = np.zeros(len(stratum_sorted), dtype=bool)
#             test_mask[::step] = True

#             for i, idx in enumerate(stratum_sorted):
#                 user = user_stats[idx]["user"]
#                 if test_mask[i]:
#                     test_users.add(user)
#                 else:
#                     train_users.add(user)

#             strata_report.append({
#                 "stratum": f"size={s_bin} fraud={f_bin}",
#                 "total"  : len(stratum_idx),
#                 "test"   : int(test_mask.sum()),
#             })

#     # ── Log des stratums ─────────────────────────────────────────────────────
#     size_labels  = ["small", "large"]
#     fraud_labels = ["low fraud", "high fraud"]
#     print(f"  Stratification des {len(unique_users)} utilisateurs :")
#     for s_bin in range(n_size_bins):
#         for f_bin in range(n_fraud_bins):
#             mask_stratum = (size_label == s_bin) & (fraud_label == f_bin)
#             n_total = mask_stratum.sum()
#             n_test  = sum(1 for u in test_users
#                           if user_stats[list(unique_users).index(u)]["user"] == u
#                           and size_label[list(unique_users).index(u)] == s_bin
#                           and fraud_label[list(unique_users).index(u)] == f_bin)
#             label = f"{size_labels[s_bin]}, {fraud_labels[f_bin]}"
#             print(f"    {label:25s} : {n_total:3d} users → {n_total - n_test:3d} train / {n_test:2d} test")

#     return train_users, test_users


# def get_dataloaders_approach_B(
#     data_dir:    str | Path,
#     train_ratio: float = 0.70,
#     val_ratio:   float = 0.10,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
# ) -> tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Approche B — Split stratifié 70/10/20 par population
#     ══════════════════════════════════════════════════════
#     Les utilisateurs sont répartis en 4 stratums (taille × fraude).
#     Dans chaque stratum, les users sont assignés de façon déterministe :
#       - 70% → TRAIN
#       - 10% → VALIDATION  (early stopping + checkpoint)
#       - 20% → TEST        (évaluation finale honnête)

#     Les 3 groupes sont 100% disjoints — aucun client n'apparaît
#     dans deux groupes différents.

#     Retourne
#     ────────
#     (train_loader, val_loader, test_loader)
#     """
#     dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
#     unique_users = np.unique(dataset.user_ids)

#     # ── Stats par user ────────────────────────────────────────────────────────
#     user_list = []
#     for user in unique_users:
#         mask  = dataset.user_ids == user
#         n_txn = int(mask.sum())
#         n_fr  = float(dataset.y[mask].sum())
#         user_list.append({"user": user, "n_txn": n_txn, "fraud_rate": n_fr / n_txn})

#     n_txns      = np.array([u["n_txn"]      for u in user_list])
#     fraud_rates = np.array([u["fraud_rate"] for u in user_list])

#     # ── Bins taille et fraude ─────────────────────────────────────────────────
#     size_bins  = np.array_split(np.argsort(n_txns),      2)
#     fraud_bins = np.array_split(np.argsort(fraud_rates), 2)
#     size_label  = np.empty(len(unique_users), dtype=int)
#     fraud_label = np.empty(len(unique_users), dtype=int)
#     for b, idx in enumerate(size_bins):  size_label[idx]  = b
#     for b, idx in enumerate(fraud_bins): fraud_label[idx] = b

#     # ── Assignation 70/10/20 dans chaque stratum ──────────────────────────────
#     train_users, val_users, test_users = set(), set(), set()

#     size_labels  = ["small", "large"]
#     fraud_labels = ["low fraud", "high fraud"]
#     print(f"  Stratification des {len(unique_users)} utilisateurs (70/10/20) :")

#     for s_bin in range(2):
#         for f_bin in range(2):
#             mask_st  = (size_label == s_bin) & (fraud_label == f_bin)
#             st_idx   = np.where(mask_st)[0]
#             if len(st_idx) == 0:
#                 continue

#             # Tri par taille → assignation déterministe cyclique
#             sorted_idx = st_idx[np.argsort(n_txns[st_idx])]
#             n          = len(sorted_idx)

#             # Calcul des points de coupure
#             n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
#             n_val  = max(1, round(n * val_ratio))
#             n_train= n - n_val - n_test

#             # Ordre : train → val → test (chronologique dans le stratum)
#             for i, idx in enumerate(sorted_idx):
#                 user = user_list[idx]["user"]
#                 if i < n_train:
#                     train_users.add(user)
#                 elif i < n_train + n_val:
#                     val_users.add(user)
#                 else:
#                     test_users.add(user)

#             label = f"{size_labels[s_bin]}, {fraud_labels[f_bin]}"
#             print(f"    {label:25s} : {n:3d} users → "
#                   f"{n_train} train / {n_val} val / {n_test} test")

#     train_indices = [i for i, u in enumerate(dataset.user_ids) if u in train_users]
#     val_indices   = [i for i, u in enumerate(dataset.user_ids) if u in val_users]
#     test_indices  = [i for i, u in enumerate(dataset.user_ids) if u in test_users]

#     _print_split_report(
#         dataset,
#         {
#             "Train (70%)": train_indices,
#             "Val   (10%)": val_indices,
#             "Test  (20%)": test_indices,
#         },
#         "Approche B — Split Stratifié 70/10/20 (taille + taux fraude)",
#         extra_info=(
#             f"Users → Train : {len(train_users):,}"
#             f"  │  Val : {len(val_users):,}"
#             f"  │  Test : {len(test_users):,}"
#             f"  │  Déterministe"
#         ),
#     )

#     # Normalisation fittée sur train uniquement
#     dataset.fit_and_apply_scaler(train_indices)

#     return (
#         _make_loader(dataset, train_indices, batch_size, shuffle=True),
#         _make_loader(dataset, val_indices,   batch_size, shuffle=False),
#         _make_loader(dataset, test_indices,  batch_size, shuffle=False),
#     )


# # ============================================================================
# # 🔄  ALIAS FL — client.py Flower
# # ============================================================================

# def get_split_dataloaders(
#     data_dir:    str | Path,
#     train_ratio: float = 0.8,
#     batch_size:  int   = 64,
#     seq_len:     int   = 5,
#     random_seed: int   = 42,
# ) -> tuple[DataLoader, DataLoader]:
#     """Alias pour client.py Flower → délègue vers Approche B."""
#     return get_dataloaders_approach_B(
#         data_dir=data_dir, train_ratio=train_ratio,
#         batch_size=batch_size, seq_len=seq_len, random_seed=random_seed,
#     )


# # ============================================================================
# # 🧪 ZONE DE TEST
# # ============================================================================

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir",   default="data/node_1/tensors")
#     parser.add_argument("--approach",   default="A", choices=["A", "B"])
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--seq_len",    type=int, default=5)
#     args = parser.parse_args()

#     if args.approach == "A":
#         train_l, val_l, test_l = get_dataloaders_approach_A(args.data_dir, batch_size=args.batch_size)
#         print(f"  Train: {len(train_l)} batches | Val: {len(val_l)} batches | Test: {len(test_l)} batches")
#         x, y = next(iter(val_l))
#         print(f"  Val X : {x.shape}  |  X range : [{x.min():.2f}, {x.max():.2f}]")
#     else:
#         train_l, test_l = get_dataloaders_approach_B(args.data_dir, batch_size=args.batch_size)
#         print(f"  Train: {len(train_l)} batches | Test: {len(test_l)} batches")








"""
dataloader.py — Pipeline de chargement de données pour la détection de fraude (FL)
====================================================================================
Stratégies de split disponibles :
  - Approche A : Split temporel INTRA-utilisateur  (70% train / 10% val / 20% test)
  - Approche B : Split par POPULATION              (80% users connus / 20% users inconnus)

Contrainte commune :
  - Séquences de longueur SEQ_LEN=5, jamais de mélange inter-utilisateurs.
  - Colonne 0 des .npy = PAN_ID (retirée avant le modèle GRU).

Normalisation :
  - RobustScaler fitté UNIQUEMENT sur les indices train + clip [-5, 5].
  - Résout le NaN causé par Time_Delta_Sec (~31M) et Velocity_Sum_24H (~2.7M).

Split 70/10/20 (Approche A) :
  - Train      : early stopping + apprentissage
  - Validation : checkpoint + early stopping (jamais utilisé pour apprendre)
  - Test       : évaluation finale UNIQUEMENT (résultat honnête, non biaisé)
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path


# ============================================================================
# 📦 DATASET
# ============================================================================

class FraudSequenceDataset(Dataset):
    def __init__(self, data_dir: str | Path, sequence_length: int = 5):
        self.data_dir        = Path(data_dir)
        self.sequence_length = sequence_length

        x_files = sorted(self.data_dir.glob("X_batch_*.npy"))
        y_files = sorted(self.data_dir.glob("y_batch_*.npy"))

        if not x_files:
            raise FileNotFoundError(f"❌ Aucun fichier X_batch_*.npy trouvé dans {self.data_dir}")
        if len(x_files) != len(y_files):
            raise ValueError(f"❌ Fichiers X ({len(x_files)}) ≠ Y ({len(y_files)})")

        all_x, all_y = [], []
        for xf, yf in zip(x_files, y_files):
            all_x.append(np.load(xf))
            all_y.append(np.load(yf))

        raw_X = np.vstack(all_x).astype(np.float32)
        raw_y = np.vstack(all_y).astype(np.float32).flatten()

        # Tri stable par PAN_ID → ordre chronologique intra-user préservé
        sort_idx      = np.argsort(raw_X[:, 0], kind="stable")
        raw_X         = raw_X[sort_idx]
        raw_y         = raw_y[sort_idx]

        self.user_ids = raw_X[:, 0]
        self.features = raw_X[:, 1:]   # (N, 26) — pas encore normalisé
        self.y        = raw_y

        n_users = len(np.unique(self.user_ids))
        print(
            f"📦 Dataset chargé : {len(self.features):,} transactions"
            f" | {n_users:,} utilisateurs uniques"
            f" | {int(self.y.sum()):,} fraudes ({100*self.y.mean():.2f} %)"
        )

    def fit_and_apply_scaler(self, train_indices: list) -> None:
        """
        RobustScaler fitté sur train uniquement + clip [-5, 5].
        Évite le data leakage et les NaN FP16.
        """
        train_features = self.features[train_indices]

        median = np.median(train_features, axis=0)
        q75, q25 = np.percentile(train_features, [75, 25], axis=0)
        iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)

        self.features = (self.features - median) / iqr
        self.features = np.clip(self.features, -5.0, 5.0)   # coupe les outliers

        print(f"📐 Normalisation appliquée (fitté sur {len(train_indices):,} txns train)")
        print(f"   Avant : min={train_features.min():.1f}  max={train_features.max():.1f}")
        print(f"   Après : min={self.features.min():.3f}  max={self.features.max():.3f}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        current_user = self.user_ids[idx]
        sequence     = []

        for lag in range(self.sequence_length - 1, -1, -1):
            target_idx = idx - lag
            if target_idx < 0 or self.user_ids[target_idx] != current_user:
                sequence.append(np.zeros(self.features.shape[1], dtype=np.float32))
            else:
                sequence.append(self.features[target_idx])

        x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
        y_tensor = torch.tensor([self.y[idx]],       dtype=torch.float32)
        return x_tensor, y_tensor


# ============================================================================
# 🔧 UTILITAIRES
# ============================================================================

def _make_loader(dataset, indices, batch_size, shuffle) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = False,
    )

def _print_split_report(dataset, splits: dict, approach_name: str, extra_info: str = ""):
    y = dataset.y
    print(f"\n{'─'*58}")
    print(f"  📊  {approach_name}")
    if extra_info:
        print(f"  {extra_info}")
    print(f"{'─'*58}")
    for name, indices in splits.items():
        n      = len(indices)
        fraud  = float(sum(y[i] for i in indices))
        pct    = 100 * fraud / max(n, 1)
        print(f"  {name:12s} │ {n:>7,} transactions │ {int(fraud):>5} fraudes ({pct:.2f} %)")
    print(f"{'─'*58}\n")


# ============================================================================
# 🅰️  APPROCHE A — Split temporel INTRA-utilisateur  70 / 10 / 20
# ============================================================================

def get_dataloaders_approach_A(
    data_dir:    str | Path,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.10,
    batch_size:  int   = 64,
    seq_len:     int   = 5,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Approche A — Split temporel 70 / 10 / 20 par utilisateur
    ══════════════════════════════════════════════════════════
    Pour chaque utilisateur (ordre chronologique) :
      - 70% premières transactions → TRAIN
      - 10% suivantes              → VALIDATION  (early stopping + checkpoint)
      - 20% dernières              → TEST        (évaluation finale honnête)

    Retourne
    ────────
    (train_loader, val_loader, test_loader)
    """
    dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
    user_ids_arr = dataset.user_ids

    # Split vectorisé O(N) — pas de boucle np.where
    change_mask  = np.concatenate(([True], user_ids_arr[1:] != user_ids_arr[:-1]))
    group_starts = np.where(change_mask)[0]
    group_ends   = np.concatenate((group_starts[1:], [len(user_ids_arr)]))
    group_sizes  = group_ends - group_starts

    train_end_pts = np.maximum(1,               (group_sizes * train_ratio).astype(int))
    val_end_pts   = np.maximum(train_end_pts+1, (group_sizes * (train_ratio + val_ratio)).astype(int))

    train_indices, val_indices, test_indices = [], [], []

    for start, size, t_end, v_end in zip(group_starts, group_sizes, train_end_pts, val_end_pts):
        idx = np.arange(start, start + size)
        train_indices.append(idx[:t_end])
        if v_end > t_end:
            val_indices.append(idx[t_end:v_end])
        if v_end < size:
            test_indices.append(idx[v_end:])

    train_indices = np.concatenate(train_indices).tolist()
    val_indices   = np.concatenate(val_indices).tolist()   if val_indices   else []
    test_indices  = np.concatenate(test_indices).tolist()  if test_indices  else []

    _print_split_report(
        dataset,
        {"Train (70%)": train_indices, "Val   (10%)": val_indices, "Test  (20%)": test_indices},
        "Approche A — Split Temporel Intra-User  70/10/20",
        extra_info=f"Users : {len(group_starts):,}",
    )

    # Normalisation fittée sur train uniquement
    dataset.fit_and_apply_scaler(train_indices)

    return (
        _make_loader(dataset, train_indices, batch_size, shuffle=True),
        _make_loader(dataset, val_indices,   batch_size, shuffle=False),
        _make_loader(dataset, test_indices,  batch_size, shuffle=False),
    )


# ============================================================================
# 🅱️  APPROCHE B — Split stratifié par taille ET taux de fraude
# ============================================================================

def _stratified_user_split(
    dataset:     FraudSequenceDataset,
    test_ratio:  float = 0.20,
    n_size_bins: int   = 2,
    n_fraud_bins:int   = 2,
) -> tuple[set, set]:
    """
    Divise les utilisateurs en train/test de façon stratifiée.

    Chaque utilisateur est classé dans un stratum selon :
      - sa taille  (nb de transactions : small / large)
      - son taux de fraude (low / high)

    On prend exactement test_ratio % de chaque stratum pour le test.
    Garantit que train et test ont la même distribution de tailles
    ET le même taux de fraude — sans dépendre d'un seed aléatoire.
    """
    unique_users = np.unique(dataset.user_ids)

    # ── Calcul des stats par user ────────────────────────────────────────────
    user_stats = []
    for user in unique_users:
        mask    = dataset.user_ids == user
        n_txn   = mask.sum()
        n_fraud = dataset.y[mask].sum()
        user_stats.append({
            "user"       : user,
            "n_txn"      : int(n_txn),
            "fraud_rate" : float(n_fraud / n_txn) if n_txn > 0 else 0.0,
        })

    n_txns      = np.array([s["n_txn"]      for s in user_stats])
    fraud_rates = np.array([s["fraud_rate"] for s in user_stats])

    # ── Création des bins ────────────────────────────────────────────────────
    # np.array_split sur les indices triés → bins de taille égale
    size_sorted  = np.argsort(n_txns)
    fraud_sorted = np.argsort(fraud_rates)

    size_bins  = np.array_split(size_sorted,  n_size_bins)
    fraud_bins = np.array_split(fraud_sorted, n_fraud_bins)

    # Assignation des bins à chaque user
    size_label  = np.empty(len(unique_users), dtype=int)
    fraud_label = np.empty(len(unique_users), dtype=int)
    for bin_idx, indices in enumerate(size_bins):
        size_label[indices]  = bin_idx
    for bin_idx, indices in enumerate(fraud_bins):
        fraud_label[indices] = bin_idx

    # ── Split test_ratio % dans chaque stratum ───────────────────────────────
    train_users : set = set()
    test_users  : set = set()

    strata_report = []
    for s_bin in range(n_size_bins):
        for f_bin in range(n_fraud_bins):
            mask_stratum = (size_label == s_bin) & (fraud_label == f_bin)
            stratum_idx  = np.where(mask_stratum)[0]

            if len(stratum_idx) == 0:
                continue

            # Tri par taille dans le stratum puis 1 sur N pour le test
            # → déterministe, pas de random
            stratum_sorted = stratum_idx[np.argsort(n_txns[stratum_idx])]
            step           = max(1, round(1.0 / test_ratio))
            test_mask      = np.zeros(len(stratum_sorted), dtype=bool)
            test_mask[::step] = True

            for i, idx in enumerate(stratum_sorted):
                user = user_stats[idx]["user"]
                if test_mask[i]:
                    test_users.add(user)
                else:
                    train_users.add(user)

            strata_report.append({
                "stratum": f"size={s_bin} fraud={f_bin}",
                "total"  : len(stratum_idx),
                "test"   : int(test_mask.sum()),
            })

    # ── Log des stratums ─────────────────────────────────────────────────────
    size_labels  = ["small", "large"]
    fraud_labels = ["low fraud", "high fraud"]
    print(f"  Stratification des {len(unique_users)} utilisateurs :")
    for s_bin in range(n_size_bins):
        for f_bin in range(n_fraud_bins):
            mask_stratum = (size_label == s_bin) & (fraud_label == f_bin)
            n_total = mask_stratum.sum()
            n_test  = sum(1 for u in test_users
                          if user_stats[list(unique_users).index(u)]["user"] == u
                          and size_label[list(unique_users).index(u)] == s_bin
                          and fraud_label[list(unique_users).index(u)] == f_bin)
            label = f"{size_labels[s_bin]}, {fraud_labels[f_bin]}"
            print(f"    {label:25s} : {n_total:3d} users → {n_total - n_test:3d} train / {n_test:2d} test")

    return train_users, test_users


def get_dataloaders_approach_B(
    data_dir:    str | Path,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.0,
    batch_size:  int   = 64,
    seq_len:     int   = 5,
) -> tuple:
    """
    Approche B — Split stratifié par population (taille × taux fraude)
    ════════════════════════════════════════════════════════════════════
    Les utilisateurs sont répartis en 4 stratums (small/large × low/high fraud).
    Assignation déterministe — sans seed aléatoire.

    Deux modes selon val_ratio :
    ────────────────────────────
    val_ratio = 0.0  (défaut) → Mode B2 : 80/20 sans validation
      - TRAIN : 80% des users → apprentissage + early stopping sur test
      - TEST  : 20% des users → évaluation finale
      → Retourne : (train_loader, test_loader)

    val_ratio > 0.0            → Mode B3 : 70/10/20 avec validation
      - TRAIN : train_ratio des users
      - VAL   : val_ratio des users   → calibration seuil + early stopping
      - TEST  : reste des users       → évaluation finale honnête
      → Retourne : (train_loader, val_loader, test_loader)
    """
    dataset      = FraudSequenceDataset(data_dir, sequence_length=seq_len)
    unique_users = np.unique(dataset.user_ids)

    # ── Stats par user ────────────────────────────────────────────────────────
    user_list = []
    for user in unique_users:
        mask  = dataset.user_ids == user
        n_txn = int(mask.sum())
        n_fr  = float(dataset.y[mask].sum())
        user_list.append({"user": user, "n_txn": n_txn, "fraud_rate": n_fr / n_txn})

    n_txns      = np.array([u["n_txn"]      for u in user_list])
    fraud_rates = np.array([u["fraud_rate"] for u in user_list])

    # ── Bins taille et fraude ─────────────────────────────────────────────────
    size_bins  = np.array_split(np.argsort(n_txns),      2)
    fraud_bins = np.array_split(np.argsort(fraud_rates), 2)
    size_label  = np.empty(len(unique_users), dtype=int)
    fraud_label = np.empty(len(unique_users), dtype=int)
    for b, idx in enumerate(size_bins):  size_label[idx]  = b
    for b, idx in enumerate(fraud_bins): fraud_label[idx] = b

    # ── Assignation 70/10/20 dans chaque stratum ──────────────────────────────
    train_users, val_users, test_users = set(), set(), set()

    size_labels  = ["small", "large"]
    fraud_labels = ["low fraud", "high fraud"]
    mode_label = f"{int(train_ratio*100)}/{int(val_ratio*100)}/{int((1-train_ratio-val_ratio)*100)}" if val_ratio > 0 else f"{int(train_ratio*100)}/{int((1-train_ratio)*100)}"
    print(f"  Stratification des {len(unique_users)} utilisateurs ({mode_label}) :")

    for s_bin in range(2):
        for f_bin in range(2):
            mask_st  = (size_label == s_bin) & (fraud_label == f_bin)
            st_idx   = np.where(mask_st)[0]
            if len(st_idx) == 0:
                continue

            # Tri par taille → assignation déterministe cyclique
            sorted_idx = st_idx[np.argsort(n_txns[st_idx])]
            n          = len(sorted_idx)

            # Calcul des points de coupure selon le mode
            if val_ratio > 0.0:
                n_test  = max(1, round(n * (1 - train_ratio - val_ratio)))
                n_val   = max(1, round(n * val_ratio))
                n_train = n - n_val - n_test
            else:
                n_test  = max(1, round(n * (1 - train_ratio)))
                n_val   = 0
                n_train = n - n_test

            # Assignation déterministe : train → val → test
            for i, idx in enumerate(sorted_idx):
                user = user_list[idx]["user"]
                if i < n_train:
                    train_users.add(user)
                elif val_ratio > 0.0 and i < n_train + n_val:
                    val_users.add(user)
                else:
                    test_users.add(user)

            label = f"{size_labels[s_bin]}, {fraud_labels[f_bin]}"
            if val_ratio > 0.0:
                print(f"    {label:25s} : {n:3d} users → {n_train} train / {n_val} val / {n_test} test")
            else:
                print(f"    {label:25s} : {n:3d} users → {n_train} train / {n_test} test")

    train_indices = [i for i, u in enumerate(dataset.user_ids) if u in train_users]
    val_indices   = [i for i, u in enumerate(dataset.user_ids) if u in val_users]
    test_indices  = [i for i, u in enumerate(dataset.user_ids) if u in test_users]

    if val_ratio > 0.0:
        splits_dict = {
            f"Train ({int(train_ratio*100)}%)": train_indices,
            f"Val   ({int(val_ratio*100)}%)":   val_indices,
            f"Test  ({int((1-train_ratio-val_ratio)*100)}%)": test_indices,
        }
        approach_label = f"Approche B — Split Stratifié {int(train_ratio*100)}/{int(val_ratio*100)}/{int((1-train_ratio-val_ratio)*100)} (taille + taux fraude)"
        extra = (
            f"Users → Train : {len(train_users):,}"
            f"  │  Val : {len(val_users):,}"
            f"  │  Test : {len(test_users):,}"
            f"  │  Déterministe"
        )
    else:
        splits_dict = {
            f"Train ({int(train_ratio*100)}%)": train_indices,
            f"Test  ({int((1-train_ratio)*100)}%)": test_indices,
        }
        approach_label = f"Approche B — Split Stratifié {int(train_ratio*100)}/{int((1-train_ratio)*100)} (taille + taux fraude)"
        extra = (
            f"Users → Train : {len(train_users):,}"
            f"  │  Test : {len(test_users):,}"
            f"  │  Déterministe"
        )

    _print_split_report(dataset, splits_dict, approach_label, extra_info=extra)

    # Normalisation fittée sur train uniquement
    dataset.fit_and_apply_scaler(train_indices)

    if val_ratio > 0.0:
        # Mode B3 — 3 loaders
        return (
            _make_loader(dataset, train_indices, batch_size, shuffle=True),
            _make_loader(dataset, val_indices,   batch_size, shuffle=False),
            _make_loader(dataset, test_indices,  batch_size, shuffle=False),
        )
    else:
        # Mode B2 — 2 loaders (sans validation)
        return (
            _make_loader(dataset, train_indices, batch_size, shuffle=True),
            _make_loader(dataset, test_indices,  batch_size, shuffle=False),
        )


# ============================================================================
# 🔄  ALIAS FL — client.py Flower
# ============================================================================

def get_split_dataloaders(
    data_dir:    str | Path,
    train_ratio: float = 0.8,
    batch_size:  int   = 64,
    seq_len:     int   = 5,
    random_seed: int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """Alias pour client.py Flower → délègue vers Approche B."""
    return get_dataloaders_approach_B(
        data_dir=data_dir, train_ratio=train_ratio,
        batch_size=batch_size, seq_len=seq_len, random_seed=random_seed,
    )


# ============================================================================
# 🧪 ZONE DE TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/node_1/tensors")
    parser.add_argument("--approach",   default="A", choices=["A", "B"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len",    type=int, default=5)
    args = parser.parse_args()

    if args.approach == "A":
        train_l, val_l, test_l = get_dataloaders_approach_A(args.data_dir, batch_size=args.batch_size)
        print(f"  Train: {len(train_l)} batches | Val: {len(val_l)} batches | Test: {len(test_l)} batches")
        x, y = next(iter(val_l))
        print(f"  Val X : {x.shape}  |  X range : [{x.min():.2f}, {x.max():.2f}]")
    else:
        train_l, test_l = get_dataloaders_approach_B(args.data_dir, batch_size=args.batch_size)
        print(f"  Train: {len(train_l)} batches | Test: {len(test_l)} batches")