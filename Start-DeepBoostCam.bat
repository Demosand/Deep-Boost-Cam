@echo off
rem ============================================================
rem  Deep-Boost-Cam launcher (venv + CUDA, Russian UI).
rem  Everything the program writes stays inside this folder:
rem  models cache, temp files, CUDA/TensorRT caches, settings.
rem  Extra arguments are passed through, e.g.:
rem     Start-DeepBoostCam.bat --lang en
rem ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

rem --- keep all user-profile / cache writes inside the project ---
set "USERPROFILE=%ROOT%\_home"
set "HOME=%ROOT%\_home"
set "APPDATA=%ROOT%\_home\AppData\Roaming"
set "LOCALAPPDATA=%ROOT%\_home\AppData\Local"
set "TEMP=%ROOT%\_temp"
set "TMP=%ROOT%\_temp"
set "CUDA_CACHE_PATH=%ROOT%\_cache\nv"
set "PIP_CACHE_DIR=%ROOT%\_cache\pip"
set "XDG_CACHE_HOME=%ROOT%\_cache"
set "TORCH_HOME=%ROOT%\_cache\torch"
set "HF_HOME=%ROOT%\_cache\huggingface"
set "PYTHONUTF8=1"

for %%D in ("_home\AppData\Roaming" "_home\AppData\Local" "_temp" "_cache\nv" "trt_cache") do (
    if not exist "%%~D" mkdir "%%~D"
)

if not exist ".venv\Scripts\python.exe" (
    echo [!] Программа ещё не установлена.
    echo     Запустите installer\Install.bat, затем снова этот файл.
    pause
    exit /b 1
)

echo Запуск Deep-Boost-Cam (CUDA)...
".venv\Scripts\python.exe" run.py --execution-provider cuda --lang ru %*
if errorlevel 1 (
    echo.
    echo [!] Программа завершилась с ошибкой. Текст ошибки выше.
    pause
)
endlocal
