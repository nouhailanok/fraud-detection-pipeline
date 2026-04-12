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

        Même les items du test_loader bénéficient de l'historique train
        via le full_dataset sous-jacent : quand __getitem__ remonte dans
        le passé (idx - i), il accède aux transactions d'entraînement du
        même user → le RNN voit bien les habitudes passées pour prédire
        les transactions futures.
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


def get_split_dataloaders(data_dir, train_ratio=0.8, batch_size=64, seq_len=5):
    """
    Divise les données du nœud en Train/Test par SPLIT TEMPOREL PAR USER.

    Pour chaque utilisateur :
        - 80% de ses transactions les plus anciennes  → Train
        - 20% de ses transactions les plus récentes   → Test

    Avantages pour la détection de fraude :
        - Le RNN est entraîné sur les habitudes passées du user
        - Le test simule une détection en conditions réelles (futur strict)
        - Les séquences du test remontent dans le passé via __getitem__
          → le contexte historique (train) est disponible pour le RNN
        - Zéro data leakage (le futur n'est jamais vu en train)
    """
    full_dataset = FraudSequenceDataset(data_dir, sequence_length=seq_len)

    # ------------------------------------------------------------------ #
    # 1. Split TEMPOREL par user                                          #
    #    Les données sont déjà triées chronologiquement grâce au         #
    #    sorted() sur les timestamps Kafka dans le Dataset                #
    # ------------------------------------------------------------------ #
    train_indices = []
    test_indices  = []

    unique_users = np.unique(full_dataset.user_ids)

    for user in unique_users:
        # Tous les index de ce user, déjà dans l'ordre chronologique
        user_indices = np.where(full_dataset.user_ids == user)[0]

        # Point de coupure temporel
        split_point = int(len(user_indices) * train_ratio)

        # Garde-fou : si un user a très peu de transactions
        if split_point == 0:
            split_point = 1  # au minimum 1 transaction en train
        if split_point >= len(user_indices):
            split_point = len(user_indices) - 1  # au minimum 1 transaction en test

        # 80% passé → train | 20% futur → test
        train_indices.extend(user_indices[:split_point].tolist())
        test_indices.extend(user_indices[split_point:].tolist())

    # ------------------------------------------------------------------ #
    # 2. Création des Subsets PyTorch                                     #
    #    Les deux Subsets partagent le même full_dataset                  #
    #    → __getitem__ peut remonter dans le passé (train) depuis le test #
    # ------------------------------------------------------------------ #
    train_ds = Subset(full_dataset, train_indices)
    test_ds  = Subset(full_dataset, test_indices)

    # ------------------------------------------------------------------ #
    # 3. DataLoaders                                                      #
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
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
    # 4. Logs de vérification                                             #
    # ------------------------------------------------------------------ #
    fraud_train = sum(full_dataset.y[i] for i in train_indices)
    fraud_test  = sum(full_dataset.y[i] for i in test_indices)

    print(f"✅ Split temporel par user (80% passé / 20% futur)")
    print(f"   Users total : {len(unique_users)}")
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