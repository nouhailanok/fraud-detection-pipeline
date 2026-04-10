import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from pathlib import Path

class FraudSequenceDataset(Dataset):
    """
    Dataset PyTorch capable de scanner un dossier de batchs npy.
    Structure attendue de X : [PAN_HASH (colonne 0), feature1, ..., feature26]
    Structure attendue de y : [is_fraud]
    """
    def __init__(self, data_dir, sequence_length=5):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        
        # 1. Scan et chargement de tous les batchs disponibles
        # sorted() est CRUCIAL pour garder l'ordre des timestamps Kafka
        x_files = sorted(list(self.data_dir.glob("X_batch_*.npy")))
        y_files = sorted(list(self.data_dir.glob("y_batch_*.npy")))
        
        if not x_files:
            raise FileNotFoundError(f"❌ Aucun fichier .npy trouvé dans {self.data_dir}")
            
        # Chargement et concaténation de tous les batchs
        all_x = []
        all_y = []
        for x_f, y_f in zip(x_files, y_files):
            all_x.append(np.load(x_f))
            all_y.append(np.load(y_f))
            
        raw_X = np.vstack(all_x).astype(np.float32)
        raw_y = np.vstack(all_y).astype(np.float32)
        
        # 2. Séparation ID (colonne 0) et Features (colonnes 1-26)
        self.user_ids = raw_X[:, 0] 
        self.features = raw_X[:, 1:] 
        self.y = raw_y.flatten()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Construit une séquence temporelle avec PADDING par utilisateur.
        L'ordre temporel est préservé à l'intérieur de chaque user.
        """
        current_user = self.user_ids[idx]
        sequence = []
        
        for i in range(self.sequence_length - 1, -1, -1):
            target_idx = idx - i
            
            # Si on dépasse le début ou si l'utilisateur change -> Padding Zéro
            if target_idx < 0 or self.user_ids[target_idx] != current_user:
                sequence.append(np.zeros(self.features.shape[1]))
            else:
                sequence.append(self.features[target_idx])
        
        x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
        y_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
        
        return x_tensor, y_tensor


def get_split_dataloaders(data_dir, train_ratio=0.8, batch_size=64, seq_len=5, random_seed=42):
    """
    Divise les données du nœud en Train/Test par UTILISATEUR.

    - 80% des users  → Train  (toutes leurs transactions, ordre temporel préservé)
    - 20% des users  → Test   (users jamais vus en train → zéro data leakage)

    L'ordre temporel des transactions est préservé à l'intérieur de chaque user
    grâce au sorted() lors du chargement des batchs Kafka.
    """
    full_dataset = FraudSequenceDataset(data_dir, sequence_length=seq_len)

    # ------------------------------------------------------------------ #
    # 1. Split des USERS (pas des transactions)                           #
    # ------------------------------------------------------------------ #
    unique_users = np.unique(full_dataset.user_ids)

    # Shuffle reproductible des users
    rng = np.random.default_rng(random_seed)
    shuffled_users = rng.permutation(unique_users)

    split_point      = int(len(shuffled_users) * train_ratio)
    train_users      = set(shuffled_users[:split_point])   # 80% des users
    test_users       = set(shuffled_users[split_point:])   # 20% des users

    # ------------------------------------------------------------------ #
    # 2. Collecte des indices de transactions par groupe                  #
    #    L'ordre est déjà chronologique grâce au sorted() du Dataset     #
    # ------------------------------------------------------------------ #
    train_indices = []
    test_indices  = []

    for idx, user in enumerate(full_dataset.user_ids):
        if user in train_users:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    # ------------------------------------------------------------------ #
    # 3. Création des Subsets PyTorch                                     #
    # ------------------------------------------------------------------ #
    train_ds = Subset(full_dataset, train_indices)
    test_ds  = Subset(full_dataset, test_indices)

    # ------------------------------------------------------------------ #
    # 4. DataLoaders                                                      #
    #    - Train : shuffle=True  → mélange les transactions inter-users  #
    #              (la séquence est auto-portée dans __getitem__)         #
    #    - Test  : shuffle=False → évaluation stable et reproductible    #
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,   # OK : chaque item porte sa propre séquence
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ------------------------------------------------------------------ #
    # 5. Logs de vérification                                             #
    # ------------------------------------------------------------------ #
    fraud_train = sum(full_dataset.y[i] for i in train_indices)
    fraud_test  = sum(full_dataset.y[i] for i in test_indices)

    print(f"✅ Split par users")
    print(f"   Users  → Train : {len(train_users)} | Test : {len(test_users)}")
    print(f"   Transactions → Train : {len(train_indices)} | Test : {len(test_indices)}")
    print(f"   Fraudes → Train : {int(fraud_train)} ({100*fraud_train/len(train_indices):.2f}%)"
          f" | Test : {int(fraud_test)} ({100*fraud_test/len(test_indices):.2f}%)")

    return train_loader, test_loader


# --- ZONE DE TEST ---
if __name__ == "__main__":
    # Exemple d'usage :
    # path = "data/node_1/tensors"
    # tr_loader, te_loader = get_split_dataloaders(path, train_ratio=0.8)
    # print(f"Batches Train : {len(tr_loader)} | Batches Test : {len(te_loader)}")
    pass