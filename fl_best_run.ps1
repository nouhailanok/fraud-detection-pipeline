# fl_best_run.ps1 — Meilleure combinaison FL
# Generé automatiquement par fl_hyperparameter_search.py
# Score : 0.9872  |  ε estimé : 0.0256

$env:FL_ROUNDS       = "15"
$env:FL_LOCAL_EPOCHS = "3"
$env:FLOWER_PORT     = "8090"
$env:FL_MIN_CLIENTS  = "4"
$env:FL_PATIENCE     = "5"
$env:FL_RESUME       = "false"
$env:DP_NOISE        = "1.2"
$env:DP_EPSILON_TARGET = "1.0"
$env:FL_LR           = "0.0005"
$env:FL_BATCH_SIZE   = "256"
# pos_weight calcule automatiquement par chaque client depuis ses donnees
