@echo off
echo.
echo Installing Qwen2.5-VL-7B API...
echo.

echo Select your GPU type:
echo 1. NVIDIA RTX 30/40 Series (RTX 3060, 3070, 3080, 3090, 4060, 4070, 4080, 4090) - CUDA 12.4
echo 2. NVIDIA RTX 20 Series / GTX 16 Series (RTX 2060-2080, GTX 1650-1660) - CUDA 12.1
echo 3. NVIDIA GTX 10 Series or older (GTX 1050-1080) - CUDA 11.8
echo 4. AMD GPU (ROCm support)
echo 5. CPU only (no GPU acceleration)
echo.
set /p choice="Enter your choice (1-5): "

echo Creating requirements_temp.txt...
if "%choice%"=="1" (
    echo --extra-index-url https://download.pytorch.org/whl/cu128/ > requirements_temp.txt
    echo Selected: NVIDIA RTX 30/40 Series - CUDA 12.8
) else if "%choice%"=="2" (
    echo --extra-index-url https://download.pytorch.org/whl/cu126 > requirements_temp.txt
    echo Selected: NVIDIA RTX 20/GTX 16 Series - CUDA 12.1
) else if "%choice%"=="3" (
    echo --extra-index-url https://download.pytorch.org/whl/cu118 > requirements_temp.txt
    echo Selected: NVIDIA GTX 10 Series - CUDA 11.8
) else if "%choice%"=="4" (
    echo --extra-index-url https://download.pytorch.org/whl/rocm6.2.4 > requirements_temp.txt
    echo Selected: AMD GPU - ROCm 6.2
) else if "%choice%"=="5" (
    echo --extra-index-url https://download.pytorch.org/whl/cpu > requirements_temp.txt
    echo Selected: CPU only
) else (
    echo Invalid choice, defaulting to RTX 30/40 Series
    exit
)

echo torch^>=2.3.0 >> requirements_temp.txt
echo torchvision^>=0.18.0 >> requirements_temp.txt
echo transformers^>=4.49.0 >> requirements_temp.txt
echo accelerate^>=0.33.0 >> requirements_temp.txt
echo bitsandbytes^>=0.43.0 >> requirements_temp.txt
echo huggingface-hub^>=0.24.0 >> requirements_temp.txt
echo fastapi^>=0.111.0 >> requirements_temp.txt
echo uvicorn^>=0.30.0 >> requirements_temp.txt
echo pillow^>=10.4.0 >> requirements_temp.txt
echo python-multipart^>=0.0.9 >> requirements_temp.txt
echo pydantic^>=2.8.0 >> requirements_temp.txt
echo qwen-vl-utils^>=0.0.8 >> requirements_temp.txt
echo psutil^>=7.0.0 >> requirements_temp.txt

echo.
echo Checking for existing virtual environment...

timeout /t 1 /nobreak >nul
if exist "env\Scripts\activate.bat" (
    echo Virtual environment already exists, skipping creation...
) else (
    echo Creating virtual environment...
    python -m venv env
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment :(
        echo Make sure Python is installed and added to PATH (USE PYTHON 3.10.11)
    )
    echo Virtual environment created successfully!
)

echo.
echo Installing Python requirements...
echo.
env\Scripts\pip.exe install --no-cache-dir -r requirements_temp.txt
if %errorlevel% neq 0 (
    echo Installation failed :(
    pause
)

echo.
echo Cleaning up temporary files...
del requirements_temp.txt

echo.
echo Creating START.bat...
echo @echo off > START.bat
echo echo Starting Qwen2.5-VL-7B API Server... >> START.bat
echo echo API Documentation: http://localhost:8000/docs >> START.bat
echo echo Base URL: http://localhost:8000 >> START.bat
echo echo GPU status URL: http://localhost:8000/gpu_status >> START.bat
echo echo Test with: curl -X POST "http://localhost:8000/analyze" -F "file=@image.jpg" >> START.bat
echo echo Press Ctrl+C to stop server >> START.bat
echo echo. >> START.bat
echo call env\Scripts\activate.bat >> START.bat
echo python AI.py >> START.bat
echo pause >> START.bat

echo.
echo Setup complete!
echo Virtual environment ready in 'env' folder
echo Run START.bat to launch the server
pause