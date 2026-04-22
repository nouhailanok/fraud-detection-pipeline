# ═══════════════════════════════════════════════════════
# federated_test_Mohamed.ps1 — Lancement FL complet
# PC medam — RTX 3050 Ti 4GB — 2 ou 4 noeuds
# ═══════════════════════════════════════════════════════

$PY  = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"
$DIR = "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"

# Verification CUDA
Write-Output "Verification CUDA sur Python 3.12..."
$cudaOk = & $PY -c "import torch; print(torch.cuda.is_available())" 2>&1
Write-Output "CUDA: $cudaOk"

if ($cudaOk -ne "True") {
    Write-Output "ERREUR: CUDA non disponible"
    Read-Host "Appuyer sur Entree"
    exit 1
}

$gpu = & $PY -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
Write-Output "GPU: $gpu"
Write-Output "Lancement FL 4 noeuds - 20 rounds x 5 epochs..."
Start-Sleep -Seconds 1

# ── SERVEUR ──────────────────────────────────────────────
# $ServerScript = {
#     Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
#     $env:PYTHONPATH                     = $PWD.Path
#     $env:FL_ROUNDS                      = "20"
#     $env:FL_MIN_CLIENTS                 = "4"
#     $env:FL_LOCAL_EPOCHS                = "5"
#     $env:FLOWER_PORT                    = "8090"
#     $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
#     $env:FL_PATIENCE                    = "5"
#     $env:FL_RESUME                      = "false"
#     Write-Output "=== SERVEUR FL Python 3.12 (4 noeuds RTX 3050 Ti) ==="
#     & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
#     Write-Output "Serveur termine"
#     Read-Host "Appuyer sur Entree pour fermer"
# }

# ── BANQUE 1 ──────────────────────────────────────────────
# $Client1Script = {
#     Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
#     $env:PYTHONPATH           = $PWD.Path
#     $env:CLIENT_ID            = "1"
#     $env:FLOWER_SERVER_HOST   = "100.71.140.30"
#     $env:FLOWER_SERVER_PORT   = "8090"
#     $env:FL_CLIENT_CONTINUOUS = "true"
#     $env:DP_NOISE             = "1.0"
#     $env:DP_EPSILON_TARGET    = "1.0"
#     $env:FL_LR                = "0.0005"
#     $env:FL_BATCH_SIZE        = "256"
#     $env:FL_POS_WEIGHT        = "100.0"
#     Write-Output "=== BANQUE 1 Python 3.12 RTX 3050 Ti ==="
#     & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
#     Write-Output "Client 1 termine"
#     Read-Host "Appuyer sur Entree pour fermer"
# }

# # ── BANQUE 2 ──────────────────────────────────────────────
# $Client2Script = {
#     Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
#     $env:PYTHONPATH           = $PWD.Path
#     $env:CLIENT_ID            = "2"
#     $env:FLOWER_SERVER_HOST   = "100.71.140.30"
#     $env:FLOWER_SERVER_PORT   = "8090"
#     $env:FL_CLIENT_CONTINUOUS = "true"
#     $env:DP_NOISE             = "1.0"
#     $env:DP_EPSILON_TARGET    = "1.0"
#     $env:FL_LR                = "0.0005"
#     $env:FL_BATCH_SIZE        = "256"
#     $env:FL_POS_WEIGHT        = "100.0"
#     Write-Output "=== BANQUE 2 Python 3.12 RTX 3050 Ti ==="
#     & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
#     Write-Output "Client 2 termine"
#     Read-Host "Appuyer sur Entree pour fermer"
# }

# ── BANQUE 3 ──────────────────────────────────────────────
$Client3Script = {
    Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
    $env:PYTHONPATH           = $PWD.Path
    $env:CLIENT_ID            = "3"
    $env:FLOWER_SERVER_HOST   = "100.71.140.30"
    $env:FLOWER_SERVER_PORT   = "8090"
    $env:FL_CLIENT_CONTINUOUS = "true"
    $env:DP_NOISE             = "1.0"
    $env:DP_EPSILON_TARGET    = "1.0"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "256"
    $env:FL_POS_WEIGHT        = "100.0"
    Write-Output "=== BANQUE 3 Python 3.12 RTX 3050 Ti ==="
    & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
    Write-Output "Client 3 termine"
    Read-Host "Appuyer sur Entree pour fermer"
}

# ── BANQUE 4 ──────────────────────────────────────────────
# $Client4Script = {
#     Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
#     $env:PYTHONPATH           = $PWD.Path
#     $env:CLIENT_ID            = "4"
#     $env:FLOWER_SERVER_HOST   = "100.71.140.30"
#     $env:FLOWER_SERVER_PORT   = "8090"
#     $env:FL_CLIENT_CONTINUOUS = "true"
#     $env:DP_NOISE             = "1.0"
#     $env:DP_EPSILON_TARGET    = "1.0"
#     $env:FL_LR                = "0.0005"
#     $env:FL_BATCH_SIZE        = "256"
#     $env:FL_POS_WEIGHT        = "100.0"
#     Write-Output "=== BANQUE 4 Python 3.12 RTX 3050 Ti ==="
#     & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
#     Write-Output "Client 4 termine"
#     Read-Host "Appuyer sur Entree pour fermer"
# }

# ── Lancement des 5 fenetres ──────────────────────────────
# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $ServerScript
Start-Sleep -Seconds 3
# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Client1Script
# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Client2Script
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Client3Script
# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Client4Script

Write-Output ""
Write-Output "5 fenetres ouvertes (1 serveur + 4 banques)"
Write-Output "Rounds: 20 - Epochs/round: 5 - Port: 8090"
Write-Output "DP target: e < 1.0 - pos_weight: 100 - RTX 3050 Ti"
