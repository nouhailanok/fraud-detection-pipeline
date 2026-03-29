import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class FraudSequenceDataset(Dataset):
    """
    Dataset PyTorch pour transformer des transactions 2D en séquences 3D (RNN/GRU).
    Structure attendue de X : [PAN_HASH (colonne 0), feature1, ..., feature26]
    Structure attendue de y : [is_fraud]
    """
    def __init__(self, x_path, y_path, sequence_length=5):
        # 1. Vérification et chargement des données
        x_path = Path(x_path)
        y_path = Path(y_path)
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Fichiers introuvables : {x_path} ou {y_path}")
            
        # Chargement en float32 (standard pour PyTorch)
        raw_X = np.load(x_path).astype(np.float32)
        raw_y = np.load(y_path).astype(np.float32)
        
        self.sequence_length = sequence_length
        
        # 2. Séparation de l'identifiant (Pivot) et des features (IA)
        # La colonne 0 est le PAN_HASH inséré lors de l'ingestion
        self.user_ids = raw_X[:, 0] 
        
        # Les colonnes 1 à 26 sont les features réelles calculées par le Vectorizer
        self.features = raw_X[:, 1:] 
        
        # Labels (is_fraud)
        self.y = raw_y

    def __len__(self):
        # Le nombre total d'exemples est égal au nombre de lignes de features
        return len(self.features)

    def __getitem__(self, idx):
        """
        Construit une séquence temporelle pour la transaction à l'index 'idx'.
        """
        current_user = self.user_ids[idx]
        sequence = []
        
        # On construit la fenêtre glissante en remontant dans le passé
        # i va de (seq_len-1) jusqu'à 0
        for i in range(self.sequence_length - 1, -1, -1):
            target_idx = idx - i
            
            # Logique de PADDING :
            # On ajoute des zéros si :
            # - On est au début du fichier (target_idx < 0)
            # - L'identifiant change (c'est un autre client/carte)
            if target_idx < 0 or self.user_ids[target_idx] != current_user:
                # Padding avec un vecteur de zéros de la taille des features (26)
                sequence.append(np.zeros(self.features.shape[1]))
            else:
                # On ajoute la transaction réelle du même utilisateur
                sequence.append(self.features[target_idx])
        
        # 3. Conversion en Tenseurs PyTorch
        # sequence devient (seq_len, 26) -> (5, 26)
        x_tensor = torch.tensor(np.stack(sequence), dtype=torch.float32)
        
        # y devient un tenseur de taille [1] pour être compatible avec les loss functions
        y_tensor = torch.tensor([self.y[idx]], dtype=torch.float32)
        
        return x_tensor, y_tensor

def get_dataloader(x_path, y_path, batch_size=64, seq_len=5, shuffle=False):
    """
    Helper pour instancier le DataLoader Flower/PyTorch.
    Note: shuffle=False est souvent préférable pour garder l'ordre temporel 
    lors de la création des fichiers, mais shuffle=True est possible ici 
    car chaque item porte déjà sa propre séquence d'historique.
    """
    dataset = FraudSequenceDataset(x_path, y_path, sequence_length=seq_len)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, # Garder à 0 sur Windows pour éviter les erreurs de multiprocessing
        pin_memory=True # Accélère le transfert vers le GPU si disponible
    )

# --- ZONE DE TEST RAPIDE ---
if __name__ == "__main__":
    # Remplace par tes vrais chemins pour tester
    # test_loader = get_dataloader("data/node_1/X_batch.npy", "data/node_1/y_batch.npy")
    # for x_seq, y_val in test_loader:
    #     print(f"Forme du batch X: {x_seq.shape}") # Doit être [64, 5, 26]
    #     print(f"Forme du batch y: {y_val.shape}") # Doit être [64, 1]
    #     break
    pass