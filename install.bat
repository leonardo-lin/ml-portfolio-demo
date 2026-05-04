@echo off
echo ============================================================
echo  QLoRA Demo Site - Installation Script
echo  Run this from demo_site directory after activating venv
echo ============================================================

echo.
echo [Step 1] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [Step 2] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [Step 3] Installing PyTorch CUDA 12.1 (compatible with CUDA 12.5)...
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

echo.
echo [Step 4] Verifying CUDA...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo [Step 5] Installing bitsandbytes (Windows native)...
pip install bitsandbytes==0.43.3

echo.
echo [Step 6] Installing training stack...
pip install transformers==4.44.2 peft==0.12.0 accelerate==0.34.2 trl==0.10.1 einops==0.8.0 datasets==2.21.0

echo.
echo [Step 7] Installing RAG stack...
pip install langchain==0.2.16 langchain-community==0.2.16 chromadb==0.5.5 sentence-transformers==3.0.1

echo.
echo [Step 8] Installing Whisper (requires ffmpeg in PATH)...
pip install setuptools
pip install openai-whisper==20231117 --no-build-isolation

echo.
echo [Step 9] Installing app and monitoring...
pip install streamlit==1.38.0 plotly==5.24.1 pynvml==11.5.0 psutil==5.9.8 python-dotenv==1.0.1

echo.
echo [Step 10] Generating experiment data...
python utils\generate_experiment_data.py

echo.
echo ============================================================
echo  Installation complete!
echo  Run: streamlit run app.py
echo ============================================================
pause
