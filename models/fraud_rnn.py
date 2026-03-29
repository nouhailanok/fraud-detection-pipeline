import torch
import torch.nn as nn

class FraudRNN(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=64):
        """
        Architecture optimisée pour le Federated Learning.
        """
        super(FraudRNN, self).__init__()
        
        # 1. LA MÉMOIRE TEMPORELLE (Le GRU)
        # On injecte directement nos 26 features. Le GRU gère la complexité.
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1,         # 1 seule couche pour rester ultra-léger
            batch_first=True
        )
        
        # 2. LE CLASSIFIEUR FINAL (La Décision)
        # Prend le résumé de 64 dimensions et sort 1 seul neurone (Logit)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x shape attendue : (Batch_Size, Sequence_Length, 26)
        """
        # --- ÉTAPE 1 : Le parcours temporel ---
        out_gru, h_n = self.gru(x)
        
        # Extraction de l'état caché FINAL
        # h_n shape : (1, Batch_Size, 64) -> squeeze -> (Batch_Size, 64)
        final_memory = h_n.squeeze(0)
        
        # --- ÉTAPE 2 : La prédiction ---
        # Pas de Sigmoid, on sort le Logit brut
        logits = self.fc_out(final_memory)
        
        return logits

# ==========================================
# --- ZONE DE TEST DU MODÈLE ---
# ==========================================
if __name__ == "__main__":
    # Batch simulé : 32 clients, 5 transactions d'historique, 26 dimensions
    dummy_batch = torch.randn(32, 5, 26) 
    
    model = FraudRNN()
    output = model(dummy_batch)
    
    # Comptage des paramètres pour ton rapport de PFE
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Forme de l'entrée X : {dummy_batch.shape}")
    print(f"Forme de la sortie y (Logits) : {output.shape}")
    print(f"Poids total du modèle à transférer en FL : {total_params} paramètres. Ultra léger ! 🚀")