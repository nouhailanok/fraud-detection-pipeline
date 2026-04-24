# Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
# $env:PYTHONPATH                     = $PWD.Path
# $env:FL_ROUNDS                      = "10"
# $env:FL_MIN_CLIENTS                 = "2"
# $env:FL_LOCAL_EPOCHS                = "5"
# $env:FLOWER_PORT                    = "8080"
# $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
# Write-Output "=== SERVEUR FL (Python 3.12) ==="
# & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py



# Set-Location "C:\Users\OsakaGamingMaroc\Downloads\fraud-detection-pipeline-main (1)\fraud-detection-pipeline-main"
# $env:PYTHONPATH = $PWD.Path
# $env:FL_ROUNDS = "20"
# $env:FL_MIN_CLIENTS = ""
# $env:FL_LOCAL_EPOCHS = "5"
# $env:FLOWER_PORT = "8080"
# $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
# $env:FL_PATIENCE = "5"
# Write-Output "=== SERVEUR FL Python 3.12 (4 noeuds) ==="
# & "C:\Users\OsakaGamingMaroc\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
# Write-Output "Serveur termine"
# Read-Host "Appuyer sur Entree pour fermer"

$PY = "C:\Users\Imane\OneDrive\Bureau\IAGI-S4\Projet_Metier\fraud-detection-pipeline\venv\Scripts\python.exe"
Write-Output "Verification CUDA sur Python 3.12..."
$cudaOk = & $PY -c "import torch; print(torch.cuda.is_available())" 2>&1
Write-Output "CUDA: $cudaOk"
if ($cudaOk -ne "True") {
    Write-Output "ERREUR CUDA non disponible"
    Read-Host "Appuyer sur Entree"
    # exit 1
}
$gpu = & $PY -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
Write-Output "GPU: $gpu"
Write-Output "Lancement FL 4 noeuds - 15 rounds x 3 epochs..."
Start-Sleep -Seconds 1

Set-Location $PSScriptRoot
$PROJECT_ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $PROJECT_ROOT

# ── Identité du run (sera incluse dans le nom du dossier) ─────────────────────
$env:NOM_USER = "Imane"   # <-- à changer

# ── Paramètres serveur / training ─────────────────────────────────────────────
# $env:PYTHONPATH = $PWD.Path changemennt du chemin relatif pour éviter les problèmes de modules
$env:PYTHONPATH = $PROJECT_ROOT
$env:FL_ROUNDS = "15"
$env:FL_MIN_CLIENTS = "4"
$env:FL_LOCAL_EPOCHS = "3"
$env:FLOWER_PORT = "8090"
$env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
$env:FL_PATIENCE = "5"

# ── OPTIONNEL : reprise depuis checkpoint (dir ou .npz)
# laisser vide = nouvel entraînement
$env:FL_RESUME_FROM = "" 
# Exemples :
# $env:FL_RESUME_FROM = "logs\runs\2026-04-18_14-32-10_nouhaila\checkpoints" 
# $env:FL_RESUME_FROM = "logs\runs\2026-04-18_14-32-10_nouhaila\checkpoints\global_model_last.npz"

if ($env:FL_RESUME_FROM -and $env:FL_RESUME_FROM.Trim().Length -gt 0) {
  $env:FL_RESUME = "true"
} else {
  $env:FL_RESUME = "false"
}

# ── Création automatique d'un dossier par run ─────────────────────────────────
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$runBase   = "logs\runs"
$runName   = "${timestamp}_$($env:NOM_USER)"
$runDir    = Join-Path $runBase $runName
$ckptDir   = Join-Path $runDir "checkpoints"

New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

# On force le serveur à écrire logs/checkpoints dans ce run
$env:FL_LOGS_DIR = $runDir
$env:FL_CHECKPOINT_DIR = $ckptDir

# ── Snapshot des variables du run + copie du script ───────────────────────────
$envFile = Join-Path $runDir "server_run_env.txt"
@(
  "NOM_USER=$env:NOM_USER",
  "FL_ROUNDS=$env:FL_ROUNDS",
  "FL_MIN_CLIENTS=$env:FL_MIN_CLIENTS",
  "FL_LOCAL_EPOCHS=$env:FL_LOCAL_EPOCHS",
  "FLOWER_PORT=$env:FLOWER_PORT",
  "FLOWER_TLS_REQUIRE_CLIENT_CERT=$env:FLOWER_TLS_REQUIRE_CLIENT_CERT",
  "FL_PATIENCE=$env:FL_PATIENCE",
  "FL_RESUME=$env:FL_RESUME",
  "FL_RESUME_FROM=$env:FL_RESUME_FROM",
  "FL_LOGS_DIR=$env:FL_LOGS_DIR",
  "FL_CHECKPOINT_DIR=$env:FL_CHECKPOINT_DIR"
) | Set-Content -Path $envFile -Encoding UTF8

Copy-Item -Path $MyInvocation.MyCommand.Path -Destination (Join-Path $runDir "server_run.ps1") -Force

Write-Output "=== SERVEUR FL | run: $runName ==="
# & "C:\Users\OsakaGamingMaroc\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
& (Get-Command python).Source federated/server.py



