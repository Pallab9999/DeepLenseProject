$ErrorActionPreference = "Stop"

Write-Host "Starting DeepLense dataset setup..."

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataRoot = Join-Path $projectRoot "Data"

# Create Data and model directories
$modelDirs = @(
    (Join-Path $dataRoot "Model_I"),
    (Join-Path $dataRoot "Model_II"),
    (Join-Path $dataRoot "Model_III")
)

New-Item -ItemType Directory -Path $dataRoot -Force | Out-Null
foreach ($dir in $modelDirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

# Install gdown in current Python env if missing
Write-Host "Checking gdown availability..."
python -m gdown --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "gdown not found. Installing with pip..."
    python -m pip install gdown
}

function Download-And-Extract {
    param(
        [string]$FileId,
        [string]$ZipPath,
        [string]$ExtractPath
    )

    Write-Host "Downloading $ZipPath ..."
    $url = "https://drive.google.com/file/d/$FileId/view?usp=sharing"
    $downloaded = $false

    # Try multiple gdown formats because some environments reject one format.
    python -m gdown $FileId -O $ZipPath
    if (Test-Path -Path $ZipPath -PathType Leaf) { $downloaded = $true }

    if (-not $downloaded) {
        python -m gdown --fuzzy $url -O $ZipPath
        if (Test-Path -Path $ZipPath -PathType Leaf) { $downloaded = $true }
    }

    if (-not $downloaded) {
        $ucUrl = "https://drive.google.com/uc?id=$FileId"
        python -m gdown $ucUrl -O $ZipPath
        if (Test-Path -Path $ZipPath -PathType Leaf) { $downloaded = $true }
    }

    if (-not $downloaded) {
        throw "Download failed: '$ZipPath' was not created. Check internet, Google Drive availability, or gdown output above."
    }

    Write-Host "Extracting to $ExtractPath ..."
    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force
}

# Model I
Download-And-Extract -FileId "1QMVLpqag6S9JWqzmGM_pK4C0F1eBVIfV" -ZipPath (Join-Path $dataRoot "Model_I\train.zip") -ExtractPath (Join-Path $dataRoot "Model_I\train")
Download-And-Extract -FileId "1rUAKLLS3p9jDaL9R9m84JVKvMcUuVsO1" -ZipPath (Join-Path $dataRoot "Model_I\test.zip")  -ExtractPath (Join-Path $dataRoot "Model_I\test")

# Model II
Download-And-Extract -FileId "1HYPkdtVUj9xsoGzFDxT4rhl37KmqDCg4" -ZipPath (Join-Path $dataRoot "Model_II\train.zip") -ExtractPath (Join-Path $dataRoot "Model_II\train")
Download-And-Extract -FileId "1PFdpqk7XOAKtg0Cnav4HTzyJiudx9dZv" -ZipPath (Join-Path $dataRoot "Model_II\test.zip")  -ExtractPath (Join-Path $dataRoot "Model_II\test")

# Model III
Download-And-Extract -FileId "1ynKMJoEeKKJqLfuKRR1Y7rQjeBMM0w94" -ZipPath (Join-Path $dataRoot "Model_III\train.zip") -ExtractPath (Join-Path $dataRoot "Model_III\train")
Download-And-Extract -FileId "18BuCv40t6qmiNnhjJF1y9rqSBhBOfDon" -ZipPath (Join-Path $dataRoot "Model_III\test.zip")  -ExtractPath (Join-Path $dataRoot "Model_III\test")

Write-Host ""
Write-Host "Done. Dataset folders are ready under: $dataRoot"
Write-Host "You can verify with: Get-ChildItem -Recurse Data\\Model_I,Data\\Model_II,Data\\Model_III"
