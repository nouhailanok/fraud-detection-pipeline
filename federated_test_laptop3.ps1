# federated_test_laptop3.ps1
# Node 3 - Se connecte au Desktop (serveur)
# IMPORTANT : remplacer SERVER_IP par l IP du Desktop (ipconfig sur Desktop)

$PY  = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"
$DIR = "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$SERVER_IP = "100.71.140.30"   # <- REMPLACER par IP du Desktop

Write-Output "Verification CUDA..."
$cudaOk = & $PY -c "import torch; print(torch.cuda.is_available())" 2>&1
Write-Output "CUDA: $cudaOk"
if ($cudaOk -ne "True") {
    Write-Output "ERREUR CUDA non disponible"
    Read-Host "Appuyer sur Entree"
    exit 1
}

$TMP = "$DIR\federated\_tmp"
New-Item -ItemType Directory -Force -Path $TMP | Out-Null

@"
Set-Location '$DIR'
`$env:PYTHONPATH = `$PWD.Path
`$env:CLIENT_ID = "3"
`$env:FLOWER_SERVER_HOST = "$SERVER_IP"
`$env:FLOWER_SERVER_PORT = "8090"
`$env:FL_CLIENT_CONTINUOUS = "true"
`$env:DP_NOISE = "1.0"
`$env:DP_EPSILON_TARGET = "1.0"
`$env:FL_LR = "0.0005"
`$env:FL_BATCH_SIZE = "256"
`$env:FL_POS_WEIGHT = "100.0"
Write-Output "BANQUE 3 - Node 3"
& '$PY' federated/client.py
Write-Output "Client 3 termine"
Read-Host "Appuyer sur Entree"
"@ | Out-File -FilePath "$TMP\client3.ps1" -Encoding UTF8

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$TMP\client3.ps1"
Write-Output "Node 3 lance - connexion vers $SERVER_IP:8090"
