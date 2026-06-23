@echo off
setlocal enabledelayedexpansion
REM ============================================================
REM  Renai Ficat 4-stage pipeline runner  (ASCII only on purpose:
REM  cmd.exe reads .bat in the OEM codepage, so non-ASCII breaks it)
REM
REM  Activate your env first, e.g.:  conda activate unet_labeling
REM
REM  Usage:
REM      run_all.bat            rebuild from existing checkpoints (default)
REM      run_all.bat rebuild    same (steps 3->4->5->6->7)
REM      run_all.bat train      retrain + rebuild (steps 2->...->7)
REM      run_all.bat full       from raw Drive images (steps 1->...->7)
REM      run_all.bat smoke      tiny sanity run (one cut, one backbone)
REM      run_all.bat web        launch the Web App only
REM ============================================================

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=rebuild"

set "PY=python"

echo ============================================================
echo  MODE = %MODE%
echo  DIR  = %CD%
echo ============================================================

REM ---- 0. env check ------------------------------------------
%PY% -c "import torch; print('[env] torch', torch.__version__, '| device =', 'cuda' if torch.cuda.is_available() else 'cpu')"
if errorlevel 1 (
  echo.
  echo [ERROR] torch not found / env not activated.
  echo         Run first:  conda activate unet_labeling
  echo.
  goto :error
)

if /i "%MODE%"=="web"     goto :web
if /i "%MODE%"=="smoke"   goto :smoke
if /i "%MODE%"=="full"    goto :step1
if /i "%MODE%"=="train"   goto :step2
if /i "%MODE%"=="rebuild" goto :step3
echo [ERROR] unknown mode: %MODE%
goto :error

:step1
echo.
echo [1/7] Prepare dataset from raw Drive images (YOLO ROI crop)...
%PY% scripts\01_prepare_stage_dataset.py ^
  --raw-root drive-download-20251023T113302Z-1-001 ^
  --weights weights\yolo_best.pt ^
  --out stage_cls_dataset ^
  --roi-csv roi_all.csv ^
  --overwrite
if errorlevel 1 goto :error

:step2
echo.
echo [2/7] Train all binary cuts (10 cuts x 5 folds x 5 backbones)... longest step
%PY% scripts\04_train_all_cuts.py
if errorlevel 1 goto :error

:step3
echo.
echo [3/7] Build OOF ensemble (voting vs stacking per cut)...
%PY% scripts\05_build_ensemble.py
if errorlevel 1 goto :error

:step4
echo.
echo [4/7] Search best hierarchy topology (selected by OOF; test = report only)...
%PY% scripts\06_search_hierarchy.py
if errorlevel 1 goto :error

:step5
echo.
echo [5/7] Train learned combiner and compare vs hierarchy (saves combiner model)...
%PY% scripts\09_train_combiner.py
if errorlevel 1 goto :error

:step6
echo.
echo [6/7] Sync Web App runtime (_runtime.json + combiner)...
%PY% scripts\07_update_web.py
if errorlevel 1 goto :error

:step7
echo.
echo [7/7] Bootstrap 95%% CIs for test metrics...
%PY% scripts\08_report_ci.py
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  DONE. Main outputs:
echo    outputs\ensemble_summary.csv
echo    outputs\hierarchy\search_results.csv      (oof + test + selected)
echo    outputs\hierarchy\best_topology.json
echo    outputs\report\ci_report.md               (with 95%% CIs)
echo    outputs\combiner\comparison.csv           (combiner vs hierarchy)
echo    web_app\_runtime.json
echo.
echo  Launch Web App:  run_all.bat web
echo ============================================================
goto :eof

:smoke
echo.
echo [smoke] sanity run (3_vs_4, efficientnet_b0, 1 fold)...
%PY% scripts\03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
if errorlevel 1 goto :error
echo [smoke] OK
goto :eof

:web
echo.
echo [web] Starting Flask:  http://127.0.0.1:5000   (Ctrl+C to stop)
%PY% web_app\app.py
goto :eof

:error
echo.
echo [FAILED] aborted in MODE=%MODE% (errorlevel %errorlevel%)
exit /b 1
