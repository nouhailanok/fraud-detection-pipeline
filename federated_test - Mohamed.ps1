# ═══════════════════════════════════════════════════════
# federated_test.ps1 — Lancement du pipeline FL local
# ═══════════════════════════════════════════════════════
# Fix : chemin Python 3.12 hardcodé DANS chaque scriptblock
# Start-Process ne transmet pas les variables externes
# ═══════════════════════════════════════════════════════

# Vérification CUDA avant de lancer
Write-Host "Verification Python 3.12 + CUDA..." -ForegroundColor DarkGray
$cudaOk = & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cudaOk -ne "True") {
    Write-Host "CUDA non disponible sur Python 3.12" -ForegroundColor Red
    Write-Host "Resultat : $cudaOk" -ForegroundColor Red
    exit 1
}
$gpuName = & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
Write-Host "OK Python 3.12 + GPU : $gpuName" -ForegroundColor Green
Start-Sleep -Seconds 1

$ProjectDir = $PWD.Path

# SERVEUR
$ServerBlock = {
    Set-Location $args[0]
    $env:PYTHONPATH                     = $args[0]
    $env:FL_ROUNDS                      = "3"
    $env:FL_MIN_CLIENTS                 = "2"
    $env:FL_LOCAL_EPOCHS                = "3"
    $env:FLOWER_PORT                    = "8080"
    $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
    Write-Host "Lancement SERVEUR Python 3.12..." -ForegroundColor Cyan
    & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
}

# BANQUE 1
$Client1Block = {
    Set-Location $args[0]
    $env:PYTHONPATH           = $args[0]
    $env:CLIENT_ID            = "1"
    $env:FLOWER_SERVER_HOST   = "127.0.0.1"
    $env:FLOWER_SERVER_PORT   = "8080"
    $env:FL_CLIENT_CONTINUOUS = "true"
    $env:DP_NOISE             = "1.5"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "256"
    $env:FL_POS_WEIGHT        = "167.0"
    Write-Host "Lancement BANQUE 1 Python 3.12 + GPU..." -ForegroundColor Green
    & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
}

# BANQUE 2
$Client2Block = {
    Set-Location $args[0]
    $env:PYTHONPATH           = $args[0]
    $env:CLIENT_ID            = "2"
    $env:FLOWER_SERVER_HOST   = "127.0.0.1"
    $env:FLOWER_SERVER_PORT   = "8080"
    $env:FL_CLIENT_CONTINUOUS = "true"
    $env:DP_NOISE             = "1.5"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "256"
    $env:FL_POS_WEIGHT        = "167.0"
    Write-Host "Lancement BANQUE 2 Python 3.12 + GPU..." -ForegroundColor Yellow
    & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServerBlock,  "-args", $ProjectDir
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client1Block, "-args", $ProjectDir
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client2Block, "-args", $ProjectDir

Write-Host ""
Write-Host "3 fenetres ouvertes (Python 3.12 + GPU)" -ForegroundColor DarkGray
Write-Host "Serveur Flower port 8080"                -ForegroundColor Cyan
Write-Host "Banque 1 node_1/tensors"                 -ForegroundColor Green
Write-Host "Banque 2 node_2/tensors"                 -ForegroundColor Yellow
Write-Host "Local epochs/round : 3"                  -ForegroundColor DarkGray
Write-Host "Logs FL : logs/fl/"                      -ForegroundColor DarkGray