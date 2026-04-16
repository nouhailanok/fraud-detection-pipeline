# ═══════════════════════════════════════════════════════
# federated_test.ps1 — Lancement du pipeline FL local
# ═══════════════════════════════════════════════════════
# Fix : force Python 3.12 (CUDA + Opacus + toutes les libs)
# au lieu de Python 3.14 (sans CUDA) utilisé par défaut.
# ═══════════════════════════════════════════════════════

# ── Python 3.12 — chemin absolu ──────────────────────────────────────────────
$PYTHON = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"

w

# Vérification rapide avant de lancer
$cudaCheck = & $PYTHON -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaCheck -ne "True") {
    Write-Host "❌ CUDA non disponible sur Python 3.12 — vérifie l'installation PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python 3.12 + CUDA OK — lancement du FL..." -ForegroundColor Green
Start-Sleep -Seconds 1

# ── SERVEUR ───────────────────────────────────────────────────────────────────
$ServerBlock = {
    param($PYTHON)
    $env:PYTHONPATH                    = $PWD.Path
    $env:FL_ROUNDS                     = "3"
    $env:FL_MIN_CLIENTS                = "2"
    $env:FLOWER_PORT                   = "8080"
    $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
    Write-Host "🚀 Lancement du SERVEUR..." -ForegroundColor Cyan
    & $PYTHON federated/server.py
}

# ── BANQUE 1 ──────────────────────────────────────────────────────────────────
$Client1Block = {
    param($PYTHON)
    $env:PYTHONPATH          = $PWD.Path
    $env:CLIENT_ID           = "1"
    $env:FLOWER_SERVER_HOST  = "127.0.0.1"
    $env:FLOWER_SERVER_PORT  = "8080"
    $env:FL_CLIENT_CONTINUOUS= "true"
    $env:DP_NOISE            = "1.5"
    $env:FL_LR               = "0.0005"
    $env:FL_BATCH_SIZE       = "256"
    $env:FL_POS_WEIGHT       = "167.0"
    Write-Host "🏦 Lancement de la BANQUE 1..." -ForegroundColor Green
    & $PYTHON federated/client.py
}

# ── BANQUE 2 ──────────────────────────────────────────────────────────────────
$Client2Block = {
    param($PYTHON)
    $env:PYTHONPATH          = $PWD.Path
    $env:CLIENT_ID           = "2"
    $env:FLOWER_SERVER_HOST  = "127.0.0.1"
    $env:FLOWER_SERVER_PORT  = "8080"
    $env:FL_CLIENT_CONTINUOUS= "true"
    $env:DP_NOISE            = "1.5"
    $env:FL_LR               = "0.0005"
    $env:FL_BATCH_SIZE       = "256"
    $env:FL_POS_WEIGHT       = "167.0"
    Write-Host "🏦 Lancement de la BANQUE 2..." -ForegroundColor Yellow
    & $PYTHON federated/client.py
}

# ── Lancement des 3 processus ─────────────────────────────────────────────────
# Serveur en premier, puis les clients après 3 secondes
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServerBlock,  "-args", $PYTHON
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client1Block, "-args", $PYTHON
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client2Block, "-args", $PYTHON

Write-Host ""
Write-Host "════════════════════════════════════════════" -ForegroundColor DarkGray
Write-Host "  3 fenêtres ouvertes :" -ForegroundColor DarkGray
Write-Host "  - Serveur Flower (port 8080)" -ForegroundColor Cyan
Write-Host "  - Banque 1 (node_1/tensors)" -ForegroundColor Green
Write-Host "  - Banque 2 (node_2/tensors)" -ForegroundColor Yellow
Write-Host "  Logs FL → logs/fl/" -ForegroundColor DarkGray
Write-Host "════════════════════════════════════════════" -ForegroundColor DarkGray