@echo off
rem ============================================================
rem  Deep-Boost-Cam - installer (Windows, NVIDIA only)
rem  Double-click it. It can be run again at any time: it only
rem  adds what is missing (use it after updating the program too).
rem  Everything is installed inside the program folder.
rem ============================================================
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0.."
set "ROOT=%CD%"
title Deep-Boost-Cam - установка

echo.
echo  ==============================================
echo     Deep-Boost-Cam - установка
echo  ==============================================
echo   Папка: %ROOT%
echo.

rem --- keep every download and cache inside the program folder ---
set "REAL_USERPROFILE=%USERPROFILE%"
set "USERPROFILE=%ROOT%\_home"
set "HOME=%ROOT%\_home"
set "TEMP=%ROOT%\_temp"
set "TMP=%ROOT%\_temp"
set "PIP_CACHE_DIR=%ROOT%\_cache\pip"
set "CUDA_CACHE_PATH=%ROOT%\_cache\nv"
set "PYTHONUTF8=1"
for %%D in ("_home" "_temp" "_cache\pip" "_cache\nv") do if not exist "%%~D" mkdir "%%~D"

rem --- 1. NVIDIA driver ---
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  [X] Не найден драйвер NVIDIA ^(nvidia-smi^).
    echo      Программа работает только на видеокартах NVIDIA.
    echo      Установите драйвер: https://www.nvidia.com/Download/index.aspx
    goto :fail
)

rem --- 2. Python 3.11 (the face-analysis library is built for 3.11) ---
set "PY="
if exist ".venv\Scripts\python.exe" goto :have_venv
py -3.11 -c "import sys" >nul 2>&1 && set "PY=py -3.11"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe""
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo  [!] Не найден Python 3.11.
    echo      Установить его сейчас через winget ^(официальный пакет Python.Python.3.11^)?
    choice /c YN /m "     Установить Python 3.11"
    if errorlevel 2 goto :nopython
    winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY="%LOCALAPPDATA%\Programs\Python\Python311\python.exe""
)
if not defined PY goto :nopython

echo  [1/2] Создаю виртуальное окружение .venv ...
%PY% -m venv .venv
if errorlevel 1 (
    echo  [X] Не удалось создать .venv
    goto :fail
)

:have_venv
echo  [2/2] Устанавливаю компоненты ^(первый раз ~6-8 ГБ загрузки^) ...
".venv\Scripts\python.exe" installer\setup_env.py %*
if errorlevel 1 goto :fail
echo.
echo  Готово. Запуск: Start-DeepBoostCam.bat ^(или ярлык на рабочем столе^).
echo.
pause
exit /b 0

:nopython
echo  [X] Нужен Python 3.11: https://www.python.org/downloads/release/python-3119/
echo      При установке отметьте "Add python.exe to PATH", затем запустите установщик снова.
:fail
echo.
echo  Установка не завершена. Сообщение об ошибке - выше.
pause
exit /b 1
