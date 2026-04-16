$PY = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"
$DIR = $PWD.Path

Write-Output "Verification CUDA..."
$cudaOk = & $PY -c "import torch; print(torch.cuda.is_available())" 2>&1
Write-Output "CUDA: $cudaOk"

if ($cudaOk -ne "True") {
    Write-Output "ERREUR CUDA non disponible"
    Read-Host "Appuyer sur Entree"
    exit 1
}

$gpu = & $PY -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
Write-Output "GPU: $gpu"
Write-Output "Lancement FL..."
Start-Sleep -Seconds 1

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\server_run.ps1"
Start-Sleep -Seconds 3
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\client1_run.ps1"
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$DIR\federated\client2_run.ps1"

Write-Output "3 fenetres ouvertes"