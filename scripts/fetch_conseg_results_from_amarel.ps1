# PowerShell Script to Download CONSeg Results from Amarel
# Usage: .\scripts\fetch_conseg_results_from_amarel.ps1

# Configuration
$AMAREL_USER = "hpl14"
$AMAREL_HOST = "amarel.rutgers.edu"
$DATASET_NAME = "Dataset777_BraTSPED2024"

# Remote paths
$REMOTE_RESULTS_DIR = "/scratch/${AMAREL_USER}/nnunet_v2_valerie/results/${DATASET_NAME}"
$REMOTE_CONSEG_OUTPUT = "${REMOTE_RESULTS_DIR}/conseg_output"
$REMOTE_CONSEG_INFERENCE = "${REMOTE_RESULTS_DIR}/conseg_inference_output"

# Local paths
$LOCAL_BASE_DIR = "conseg_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $LOCAL_BASE_DIR | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Downloading CONSeg Results from Amarel" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Remote server: ${AMAREL_USER}@${AMAREL_HOST}" -ForegroundColor Yellow
Write-Host "Dataset: ${DATASET_NAME}" -ForegroundColor Yellow
Write-Host "Local directory: ${LOCAL_BASE_DIR}" -ForegroundColor Yellow
Write-Host ""

# Function to download with rsync-style progress
function Download-Directory {
    param(
        [string]$RemotePath,
        [string]$LocalPath,
        [string]$Description
    )
    
    Write-Host "Downloading ${Description}..." -ForegroundColor Green
    Write-Host "  From: ${RemotePath}" -ForegroundColor Gray
    Write-Host "  To: ${LocalPath}" -ForegroundColor Gray
    
    New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
    
    $scpCommand = "scp -r ${AMAREL_USER}@${AMAREL_HOST}:${RemotePath}/* ${LocalPath}/"
    Write-Host "  Running: $scpCommand" -ForegroundColor Gray
    
    Invoke-Expression $scpCommand
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  SUCCESS: Downloaded ${Description}" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Failed to download ${Description}" -ForegroundColor Red
    }
    Write-Host ""
}

# 1. Download calibration data
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 1: Downloading Calibration Data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$calibrationFile = "${REMOTE_CONSEG_OUTPUT}/calibration_data.json"
$localCalibration = "${LOCAL_BASE_DIR}/calibration"
New-Item -ItemType Directory -Force -Path $localCalibration | Out-Null

scp "${AMAREL_USER}@${AMAREL_HOST}:${calibrationFile}" "${localCalibration}/"
Write-Host ""

# 2. Download inference results
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 2: Downloading Inference Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Download-Directory `
    -RemotePath "${REMOTE_CONSEG_INFERENCE}" `
    -LocalPath "${LOCAL_BASE_DIR}/inference" `
    -Description "Segmentation predictions"

# 3. Download uncertainty maps
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 3: Downloading Uncertainty Maps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Download-Directory `
    -RemotePath "${REMOTE_CONSEG_INFERENCE}/uncertainty_maps" `
    -LocalPath "${LOCAL_BASE_DIR}/uncertainty_maps" `
    -Description "Uncertainty maps"

# 4. Download uncertainty statistics
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 4: Downloading Uncertainty Statistics" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Download-Directory `
    -RemotePath "${REMOTE_CONSEG_INFERENCE}/uncertainty_stats" `
    -LocalPath "${LOCAL_BASE_DIR}/uncertainty_stats" `
    -Description "Uncertainty statistics"

# 5. Download correlation analysis
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 5: Downloading Correlation Analysis" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$correlationFile = "${REMOTE_CONSEG_INFERENCE}/uncertainty_dice_correlation.json"
scp "${AMAREL_USER}@${AMAREL_HOST}:${correlationFile}" "${LOCAL_BASE_DIR}/" 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "  SUCCESS: Downloaded correlation analysis" -ForegroundColor Green
} else {
    Write-Host "  INFO: Correlation analysis not available (requires ground truth)" -ForegroundColor Yellow
}
Write-Host ""

# 6. Download log files
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 6: Downloading Log Files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$logsDir = "${LOCAL_BASE_DIR}/logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# Get recent log files
scp "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/conseg_*.log" "${logsDir}/" 2>$null
scp "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/conseg_*.err" "${logsDir}/" 2>$null

Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Download Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "All results downloaded to: ${LOCAL_BASE_DIR}" -ForegroundColor Green
Write-Host ""
Write-Host "Directory structure:" -ForegroundColor Yellow
Write-Host "  ${LOCAL_BASE_DIR}/" -ForegroundColor Gray
Write-Host "    calibration/               - Calibration data and threshold" -ForegroundColor Gray
Write-Host "    inference/                 - Segmentation predictions" -ForegroundColor Gray
Write-Host "    uncertainty_maps/          - Voxel-wise uncertainty maps" -ForegroundColor Gray
Write-Host "    uncertainty_stats/         - Per-case uncertainty statistics" -ForegroundColor Gray
Write-Host "    uncertainty_dice_correlation.json - Correlation analysis" -ForegroundColor Gray
Write-Host "    logs/                      - Job logs and errors" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Generate report: python generate_conseg_report.py --results_dir ${LOCAL_BASE_DIR}" -ForegroundColor White
Write-Host "  2. Visualize results: python visualize_conseg_results.py --results_dir ${LOCAL_BASE_DIR}" -ForegroundColor White
Write-Host ""
