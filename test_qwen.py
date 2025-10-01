from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

# РїСѓС‚СЊ РґР»СЏ СЃРѕС…СЂР°РЅРµРЅРёСЏ РјРѕРґРµР»Рё
model_dir = r"C:\Users\O.Iunina\Desktop\Projects\RAG\models"

# СЃРєР°С‡РёРІР°РЅРёРµ РјРѕРґРµР»Рё (СЃСѓС‰РµСЃС‚РІСѓСЋС‰Р°СЏ РІРµСЂСЃРёСЏ Qwen-14B 4bit GGUF)
model_file = hf_hub_download(
    repo_id="TheBloke/Qwen-14B-4bit-GGUF",
    filename="qwen-14b.Q4_K_M.gguf",
    cache_dir=model_dir
)

# РёРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РјРѕРґРµР»Рё
llm = Llama(model_path=model_file, n_threads=32)

# С‚РµСЃС‚РѕРІС‹Р№ Р·Р°РїСЂРѕСЃ
prompt = "РћР±СЉСЏСЃРЅРё, С‡С‚Рѕ С‚Р°РєРѕРµ Windows Server 2019, РєРѕСЂРѕС‚РєРѕ Рё РїРѕ РґРµР»Сѓ."
response = llm(prompt, max_tokens=256)
print(response['choices'][0]['text'])
