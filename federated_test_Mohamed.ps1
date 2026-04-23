# $PY  = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"
# $DIR = "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"

# Write-Output "Verification CUDA sur Python 3.12..."
# $cudaOk = & $PY -c "import torch; print(torch.cuda.is_available())" 2>&1
# Write-Output "CUDA: $cudaOk"

# if ($cudaOk -ne "True") {
#     Write-Output "ERREUR: CUDA non disponible"
#     Read-Host "Appuyer sur Entree"
#     exit 1
# }

# $gpu = & $PY -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
# Write-Output "GPU: $gpu"
# Write-Output "Lancement FL 2 noeuds - 10 rounds x 5 epochs..."
# Start-Sleep -Seconds 1

# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$DIR\federated\server_run.ps1`""
# Start-Sleep -Seconds 3
# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$DIR\federated\client1_run.ps1`""
# Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$DIR\federated\client2_run.ps1`""

# Write-Output "3 fenetres ouvertes (1 serveur + 2 banques)"
# Write-Output "Rounds: 10 - Epochs/round: 5 - Port: 8090"


$PY  = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"
$DIR = "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"

Write-Output "Verification CUDA sur Python 3.12..."
$cudaOk = & $PY -c "import torch; print(torch.cuda.is_available())" 2>&1
Write-Output "CUDA: $cudaOk"
if ($cudaOk -ne "True") {
    Write-Output "ERREUR CUDA non disponible"
    Read-Host "Appuyer sur Entree"
    exit 1
}
$gpu = & $PY -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
Write-Output "GPU: $gpu"
Write-Output "Lancement FL 4 noeuds - 15 rounds x 3 epochs..."
Start-Sleep -Seconds 1

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\server_run.ps1"
Start-Sleep -Seconds 4
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\client1_run.ps1"
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\client2_run.ps1"
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\client3_run.ps1"
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\client4_run.ps1"

Write-Output "5 fenetres ouvertes - 15 rounds x 3 epochs"