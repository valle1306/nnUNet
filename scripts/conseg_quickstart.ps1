# CONSeg Quick Start Script for Amarel
# This script automates the entire CONSeg workflow on Amarel

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CONSeg Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$AMAREL_USER = "hpl14"
$AMAREL_HOST = "amarel.rutgers.edu"

# Step 1: Prepare calibration set locally
Write-Host "Step 1: Preparing calibration set..." -ForegroundColor Yellow
Write-Host ""

conda activate nnunetv2
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate conda environment" -ForegroundColor Red
    exit 1
}

$env:nnUNet_raw = "C:\path\to\nnUNet_raw"  # UPDATE THIS

python prepare_calibration_set.py `
    --dataset Dataset777_BraTSPED2024 `
    --calibration_ratio 0.10 `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Calibration set preparation failed" -ForegroundColor Red
    exit 1
}

Write-Host "SUCCESS: Calibration set prepared" -ForegroundColor Green
Write-Host ""

# Step 2: Upload to Amarel
Write-Host "Step 2: Uploading to Amarel..." -ForegroundColor Yellow
Write-Host ""

$DATASET_DIR = "${env:nnUNet_raw}\Dataset777_BraTSPED2024"
$REMOTE_DATASET = "/scratch/${AMAREL_USER}/nnunet_v2_valerie/raw/Dataset777_BraTSPED2024"

# Upload calibration data
Write-Host "Uploading calibration images..." -ForegroundColor Gray
scp -r "${DATASET_DIR}\imagesTr_calibration" "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_DATASET}/"

Write-Host "Uploading calibration labels..." -ForegroundColor Gray
scp -r "${DATASET_DIR}\labelsTr_calibration" "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_DATASET}/"

Write-Host "Uploading split info..." -ForegroundColor Gray
scp "${DATASET_DIR}\conseg_calibration_split.json" "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_DATASET}/"

# Upload scripts
Write-Host "Uploading CONSeg scripts..." -ForegroundColor Gray
scp run_conseg_calibration.py "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/"
scp run_conseg_inference.py "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/"
scp conseg_calibration_brats_amarel.sbatch "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/"
scp conseg_inference_brats_amarel.sbatch "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/"
scp nnunetv2\inference\predict_with_conseg.py "${AMAREL_USER}@${AMAREL_HOST}:~/nnUNet/nnunetv2/inference/"

Write-Host "SUCCESS: Upload complete" -ForegroundColor Green
Write-Host ""

# Step 3: Submit calibration job
Write-Host "Step 3: Submitting calibration job..." -ForegroundColor Yellow
Write-Host ""

ssh "${AMAREL_USER}@${AMAREL_HOST}" "cd ~/nnUNet && sbatch conseg_calibration_brats_amarel.sbatch"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to submit calibration job" -ForegroundColor Red
    exit 1
}

Write-Host "SUCCESS: Calibration job submitted" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Monitor calibration job:" -ForegroundColor White
Write-Host "   ssh ${AMAREL_USER}@${AMAREL_HOST}" -ForegroundColor Gray
Write-Host "   squeue -u ${AMAREL_USER}" -ForegroundColor Gray
Write-Host "   tail -f ~/nnUNet/conseg_calibration_*.log" -ForegroundColor Gray
Write-Host ""
Write-Host "2. After calibration completes, submit inference job:" -ForegroundColor White
Write-Host "   ssh ${AMAREL_USER}@${AMAREL_HOST}" -ForegroundColor Gray
Write-Host "   cd ~/nnUNet" -ForegroundColor Gray
Write-Host "   sbatch conseg_inference_brats_amarel.sbatch" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Download results:" -ForegroundColor White
Write-Host "   .\scripts\fetch_conseg_results_from_amarel.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Generate report:" -ForegroundColor White
Write-Host "   python generate_conseg_report.py --results_dir conseg_results_YYYYMMDD_HHMMSS" -ForegroundColor Gray
Write-Host ""
