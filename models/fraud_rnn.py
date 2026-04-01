import torch
import torch.nn as nn

class FraudRNN(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, num_layers=2, dropout_rate=0.2):
        """
        Version finale pour le projet FL : Profondeur accrue et régularisation.
        """
        super(FraudRNN, self).__init__()
        
        # 1. LA MÉMOIRE (Stacked GRU)
        # On passe à 2 couches et on intègre le dropout entre ces couches
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,        
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0  # Sécurité : le dropout exige au moins 2 couches
        )
        
        # 2. LE CLASSIFIEUR ROBUSTE (MLP)
        # Au lieu de passer de 128 à 1 direct, on fait une transition douce
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),     # Réduction de 128 à 64
            nn.ReLU(),                     # Activation non-linéaire (donne de la souplesse)
            nn.Dropout(dropout_rate),      # Désactive 20% des neurones pour éviter le surapprentissage
            nn.Linear(64, 1)               # Sortie finale (Logit)
        )

    def forward(self, x):
        """
        x shape attendue : (Batch_Size, Sequence_Length, 26)
        """
        out_gru, h_n = self.gru(x)
        
        # ATTENTION AU CHANGEMENT ICI !
        # Avec num_layers=2, h_n a la forme : (2, Batch_Size, 128)
        # On ne veut extraire que la mémoire de la TOUTE DERNIÈRE couche du GRU.
        # En Python, l'indice -1 permet de prendre le dernier élément.
        final_memory = h_n[-1, :, :]  # Shape finale : (Batch_Size, 128)
        
        # La prédiction via notre nouveau classifieur
        logits = self.classifier(final_memory)
        
        return logits

# ==========================================
# --- ZONE DE TEST DU MODÈLE ---
# ==========================================
if __name__ == "__main__":
    # Test avec un batch de 32 clients, 5 transactions d'historique, 26 features
    dummy_batch = torch.randn(32, 5, 26) 
    
    model = FraudRNN(hidden_dim=128, num_layers=2, dropout_rate=0.2)
    output = model(dummy_batch)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Forme de l'entrée X : {dummy_batch.shape}")
    print(f"Forme de la sortie y (Logits) : {output.shape}")
    print(f"Total des paramètres à synchroniser en FL : {total_params:,}")



# import torch
# import torch.nn as nn

# class FraudRNN(nn.Module):
#     def __init__(self, input_dim=26, hidden_dim=64):
#         """
#         Architecture optimisée pour le Federated Learning.
#         """
#         super(FraudRNN, self).__init__()
        
#         # 1. LA MÉMOIRE TEMPORELLE (Le GRU)
#         # On injecte directement nos 26 features. Le GRU gère la complexité.
#         self.gru = nn.GRU(
#             input_size=input_dim, 
#             hidden_size=hidden_dim, 
#             num_layers=1,         # 1 seule couche pour rester ultra-léger
#             batch_first=True
#         )
        
#         # 2. LE CLASSIFIEUR FINAL (La Décision)
#         # Prend le résumé de 64 dimensions et sort 1 seul neurone (Logit)
#         self.fc_out = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         """
#         x shape attendue : (Batch_Size, Sequence_Length, 26)
#         """
#         # --- ÉTAPE 1 : Le parcours temporel ---
#         out_gru, h_n = self.gru(x)
        
#         # Extraction de l'état caché FINAL
#         # h_n shape : (1, Batch_Size, 64) -> squeeze -> (Batch_Size, 64)
#         final_memory = h_n.squeeze(0)
        
#         # --- ÉTAPE 2 : La prédiction ---
#         # Pas de Sigmoid, on sort le Logit brut
#         logits = self.fc_out(final_memory)
        
#         return logits

# # ==========================================
# # --- ZONE DE TEST DU MODÈLE ---
# # ==========================================
# if __name__ == "__main__":
#     # Batch simulé : 32 clients, 5 transactions d'historique, 26 dimensions
#     dummy_batch = torch.randn(32, 5, 26) 
    
#     model = FraudRNN()
#     output = model(dummy_batch)
    
#     # Comptage des paramètres pour ton rapport de PFE
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"Forme de l'entrée X : {dummy_batch.shape}")
#     print(f"Forme de la sortie y (Logits) : {output.shape}")
#     print(f"Poids total du modèle à transférer en FL : {total_params} paramètres. Ultra léger ! 🚀")