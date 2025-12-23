# CONSeg Results Download Script
# Run this to download all results from Amarel

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "CONSeg Results Download" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Create directory structure
$baseDir = "c:\Users\lpnhu\Downloads\nnUNet\conseg_results_complete"
Write-Host "Creating directory: $baseDir" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "$baseDir\segmentations" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\uncertainty_maps" | Out-Null
New-Item -ItemType Directory -Force -Path "$baseDir\stats" | Out-Null

$amarelPath = "/scratch/hpl14/nnunet_v2_valerie/results/Dataset777_BraTSPED2024/conseg_output"
$user = "hpl14@amarel.rutgers.edu"

# Download summary
Write-Host ""
Write-Host "Downloading summary..." -ForegroundColor Cyan
scp "${user}:${amarelPath}/summary.json" "$baseDir\"

# Download segmentations (52 files, ~500MB)
Write-Host ""
Write-Host "Downloading segmentations (52 files)..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
scp -r "${user}:${amarelPath}/segmentations/*" "$baseDir\segmentations\"

# Download uncertainty maps (52 files, ~500MB)
Write-Host ""
Write-Host "Downloading uncertainty maps (52 files)..." -ForegroundColor Cyan
scp -r "${user}:${amarelPath}/uncertainty_maps/*" "$baseDir\uncertainty_maps\"

# Download statistics (52 JSON files, <1MB)
Write-Host ""
Write-Host "Downloading statistics (52 files)..." -ForegroundColor Cyan
scp -r "${user}:${amarelPath}/stats/*" "$baseDir\stats\"

# Summary
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Download Complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Yellow
Write-Host "  $baseDir" -ForegroundColor White
Write-Host ""
Write-Host "Contents:" -ForegroundColor Yellow
$segCount = (Get-ChildItem "$baseDir\segmentations" -Filter "*.nii.gz" -ErrorAction SilentlyContinue).Count
$uncCount = (Get-ChildItem "$baseDir\uncertainty_maps" -Filter "*.nii.gz" -ErrorAction SilentlyContinue).Count
$statCount = (Get-ChildItem "$baseDir\stats" -Filter "*.json" -ErrorAction SilentlyContinue).Count

Write-Host "  Segmentations: $segCount files" -ForegroundColor White
Write-Host "  Uncertainty maps: $uncCount files" -ForegroundColor White
Write-Host "  Statistics: $statCount files" -ForegroundColor White
Write-Host ""

# Display summary.json
if (Test-Path "$baseDir\summary.json") {
    Write-Host "Summary Statistics:" -ForegroundColor Yellow
    $summary = Get-Content "$baseDir\summary.json" | ConvertFrom-Json
    Write-Host "  Cases processed: $($summary.num_cases_processed)/$($summary.num_cases_total)" -ForegroundColor White
    Write-Host "  Mean Uncertainty Ratio: $([math]::Round($summary.uncertainty_ratio_statistics.mean * 100, 4))%" -ForegroundColor White
    Write-Host "  Median Uncertainty Ratio: $([math]::Round($summary.uncertainty_ratio_statistics.median * 100, 4))%" -ForegroundColor White
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open results in ITK-SNAP or 3D Slicer for visualization" -ForegroundColor Gray
Write-Host "  2. Analyze per-case statistics in stats/*.json files" -ForegroundColor Gray
Write-Host "  3. See CONSEG_COMPLETION_REPORT.md for full details" -ForegroundColor Gray
Write-Host ""
