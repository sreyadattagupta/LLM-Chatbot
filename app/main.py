from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

app = FastAPI()

BASE_MODEL = "meta-llama/Llama-3.1-8B"  # or your specific base
ADAPTER_PATH = "./adapter"              # folder with adapter_model.safetensors

print("Loading base model…")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Applying LoRA adapter…")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": text}
