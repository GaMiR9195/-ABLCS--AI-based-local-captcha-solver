import os
from datetime import datetime
import torch
import json
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import psutil
from huggingface_hub import snapshot_download
import uvicorn
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from qwen_vl_utils import process_vision_info

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    custom_prompt: Optional[str] = None

class QwenGPUAPI:
    def __init__(self, model_dir="./qwen2.5-vl-7b-instruct"):
        self.model_dir = Path(model_dir)
        self.prompt_file = Path("./prompt.txt")
        
        self.cached_prompt = None
        self.prompt_last_modified = None
        self.prompt_last_checked = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")
        
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f}GB")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        self._ensure_model_downloaded()
        self._load_model()
        
    def _ensure_model_downloaded(self):
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            print("DON'T LET IT DOWNLOAD... Downloading Qwen2.5-VL-7B...")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
    
    def _load_model(self):
        print("Loading model with 8-bit quantization and CPU+GPU distribution...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))

        # Исправление: копируем chat_template из tokenizer в processor
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            self.processor.chat_template = self.tokenizer.chat_template

        offload_folder = self.model_dir / "offload_cache"
        offload_folder.mkdir(exist_ok=True)

        import psutil
        cpu_memory = psutil.virtual_memory().total
        gpu_memory = torch.cuda.get_device_properties(0).total_memory

        gpu_limit_gb = round(gpu_memory * 0.65 / (1024**3), 1)
        cpu_limit_gb = round(cpu_memory * 0.50 / (1024**3), 1)

        print(f"GPU memory limit: {gpu_limit_gb}GB of {gpu_memory/(1024**3):.1f}GB")
        print(f"CPU memory limit: {cpu_limit_gb}GB of {cpu_memory/(1024**3):.1f}GB")
        print(f"Offload folder: {offload_folder}")

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={
                0: f"{gpu_limit_gb}GB",
                "cpu": f"{cpu_limit_gb}GB"
            },
            offload_folder=str(offload_folder),
            offload_state_dict=True,
            trust_remote_code=True
        ).eval()

        print("\nModel distribution:")
        if hasattr(self.model, 'hf_device_map'):
            gpu_layers = sum(1 for d in self.model.hf_device_map.values() if d == 0)
            cpu_layers = sum(1 for d in self.model.hf_device_map.values() if d == 'cpu')
            disk_layers = sum(1 for d in self.model.hf_device_map.values() if d == 'disk')
            print(f"  Layers on GPU: {gpu_layers}")
            print(f"  Layers on CPU: {cpu_layers}")
            print(f"  Layers on Disk: {disk_layers}")

        print(f"\nActual GPU memory used: {torch.cuda.memory_allocated()/(1024**3):.2f}GB")
        print(f"Actual CPU memory used: {psutil.Process().memory_info().rss/(1024**3):.2f}GB")

        offload_size = sum(f.stat().st_size for f in offload_folder.glob('*') if f.is_file())
        if offload_size > 0:
            print(f"Offload folder size: {offload_size/(1024**3):.2f}GB")

        torch.cuda.empty_cache()
        print("Model loaded successfully with 8-bit quantization!")
    
    def _load_prompt(self):
        if self.prompt_file.exists():
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        default_prompt = "Extract all text from this image accurately and completely."
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            f.write(default_prompt)
        return default_prompt

    def _get_prompt(self):
        """Получает промпт с проверкой изменений файла"""
        # Проверяем не чаще раза в секунду
        now = datetime.now()
        if self.prompt_last_checked and (now - self.prompt_last_checked).seconds < 1:
            return self.cached_prompt
        
        self.prompt_last_checked = now
        
        try:
            current_mtime = os.path.getmtime(self.prompt_file)
            
            # Если файл изменился или первая загрузка
            if self.prompt_last_modified != current_mtime:
                with open(self.prompt_file, 'r', encoding='utf-8') as f:
                    self.cached_prompt = f.read().strip()
                self.prompt_last_modified = current_mtime
                print(f"Prompt reloaded: {self.cached_prompt[:50]}...")
                
        except FileNotFoundError:
            # Создаем файл с дефолтным промптом
            self.cached_prompt = "Extract all text from this image accurately and completely."
            with open(self.prompt_file, 'w', encoding='utf-8') as f:
                f.write(self.cached_prompt)
            self.prompt_last_modified = os.path.getmtime(self.prompt_file)
        
        return self.cached_prompt

    def _decode_image(self, image_data):
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        else:
            image = Image.open(BytesIO(image_data))
        return image.convert('RGB')
    
    def _process_image_sync(self, image_data, custom_prompt):
        try:
            image = self._decode_image(image_data)
            prompt = custom_prompt if custom_prompt else self._get_prompt()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(self.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]

            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

            return {
                "success": True,
                "result": response,
                "prompt_used": prompt,
                "device": str(self.device)
            }

        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "device": str(self.device)
            }
    
    async def analyze_image_async(self, image_data, custom_prompt=None):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_image_sync,
            image_data,
            custom_prompt
        )
        return result

qwen_api = QwenGPUAPI()
app = FastAPI(title="Qwen2.5-VL GPU API", version="2.0.0")

@app.get("/")
async def root():
    return {
        "message": "Qwen2.5-VL GPU API Server",
        "status": "running",
        "device": str(qwen_api.device),
        "gpu": torch.cuda.get_device_name(0),
        "memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
        "endpoints": {
            "POST /analyze": "Upload image file",
            "POST /analyze_base64": "Base64 image",
            "GET /prompt": "Get prompt",
            "PUT /prompt": "Update prompt"
        }
    }

@app.post("/analyze")
async def analyze_image_upload(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    image_data = await file.read()
    result = await qwen_api.analyze_image_async(image_data, custom_prompt)
    
    return JSONResponse(content=result)

@app.post("/analyze_base64")
async def analyze_image_base64(request: ImageAnalysisRequest):
    result = await qwen_api.analyze_image_async(request.image_base64, request.custom_prompt)
    return JSONResponse(content=result)

@app.get("/prompt")
async def get_prompt():
    prompt = qwen_api._load_prompt()
    return {"current_prompt": prompt}

@app.put("/prompt")
async def update_prompt(new_prompt: str = Form(...)):
    try:
        with open(qwen_api.prompt_file, 'w', encoding='utf-8') as f:
            f.write(new_prompt)
        return {"success": True, "message": "Prompt updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": qwen_api.model is not None,
        "device": str(qwen_api.device),
        "gpu_memory": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
    }

@app.get("/gpu_status")
async def gpu_status():
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB",
        "memory_reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB",
        "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )