@echo off
rem Double-click: continue (or start) White Stake training in its own minimised window, then open Claude Code here.
rem  - Progress is saved after every iteration (checkpoint, optimizer, schedule, log). If training was stopped -
rem    window closed, PC restarted - the newest unfinished run in checkpoints\runs is resumed from its last
rem    saved iteration. A run is finished when it has az_done.txt; then a new run folder is started.
rem  - If a run is already going, it doesn't start another; it just opens Claude Code.
rem  - Closing Claude Code does not stop training; closing the "Balatro training" window does (and is safe).

cd /d "%~dp0"

powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*balatro_rl.az.train run*' }) { exit 1 } else { exit 0 }"
if errorlevel 1 goto running

set RUN=
for /f "delims=" %%d in ('powershell -NoProfile -Command "$r = Get-ChildItem checkpoints\runs -Directory -ErrorAction SilentlyContinue | Where-Object { (Test-Path (Join-Path $_.FullName 'az.pt')) -and -not (Test-Path (Join-Path $_.FullName 'az_done.txt')) } | Sort-Object Name | Select-Object -Last 1; if ($r) { $r.Name }"') do set RUN=%%d
if defined RUN (
    set WHAT=Resuming
) else (
    set WHAT=Starting
    for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set RUN=az_%%t
)
set OUT=checkpoints\runs\%RUN%
mkdir "%OUT%" 2>nul

start "Balatro training" /min cmd /c "python -m balatro_rl.az.train run --resume --iters 50 --games 64 --workers 7 --eval-every 5 --eval-games 100 --out %OUT%\az.pt --data %OUT%\data >> %OUT%\stdout.txt 2>> %OUT%\stderr.txt"

echo.
echo %WHAT% training in %OUT%
echo   progress:  python -m balatro_rl.az.report --log %OUT%\az_log.jsonl
echo   errors:    %OUT%\stderr.txt
echo.
goto claude

:running
echo.
echo A training run is already going - not starting another one.
echo.

:claude
claude
