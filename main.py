# /home/owner/Python/apiTranslate/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import traceback
import json

load_dotenv()

app = FastAPI()
HF_TOKEN = os.getenv("HF_TOKEN")

# InferenceClient-ийг шинэ endpoint-тэй үүсгэх
def get_inference_client() :
    return InferenceClient(
        token=HF_TOKEN,
        # base_url="https://router.huggingface.co/huggingface"
        base_url="https://xxx-xxx.us-east-1.aws.endpoints.huggingface.cloud"
    )

class TranslateRequest(BaseModel):
    text: str
    src_lang: str = "mn_MN"
    tgt_lang: str = "en_XX"
    model: str = "facebook/mbart-large-50-many-to-many-mmt"

def get_nllb_lang_code(lang_code: str) -> str:
    """NLLB model-д зориулсан хэлний код хөрвүүлэх"""
    lang_code = lang_code.lower()
    
    if lang_code in ['mn', 'mn_mn', 'mon', 'mongolian', 'mn_cyrl']:
        return "mon_Cyrl"
    elif lang_code in ['mn_mong', 'mon_mong', 'mong']:
        return "mon_Mong"
    elif lang_code in ['en', 'en_xx', 'eng', 'english']:
        return "eng_Latn"
    elif lang_code in ['ru', 'ru_ru', 'rus', 'russian']:
        return "rus_Cyrl"
    elif lang_code in ['zh', 'zh_cn', 'zho', 'chinese']:
        return "zho_Hans"
    
    return lang_code

def get_lang_codes_for_model(model: str, src_lang: str, tgt_lang: str):
    """Model-д тохирсон хэлний кодыг буцаана"""
    model_lower = model.lower()
    
    if "nllb" in model_lower:
        return get_nllb_lang_code(src_lang), get_nllb_lang_code(tgt_lang)
    elif "mbart" in model_lower:
        src = "mn_MN" if src_lang in ["mn", "mn_MN", "mon", "mn_cyrl"] else src_lang
        tgt = "en_XX" if tgt_lang in ["en", "en_XX", "eng"] else tgt_lang
        return src, tgt
    elif "m2m100" in model_lower:
        src = src_lang.split("_")[0] if "_" in src_lang else src_lang
        tgt = tgt_lang.split("_")[0] if "_" in tgt_lang else tgt_lang
        return src, tgt
    else:
        return src_lang, tgt_lang

@app.post("/translate")
async def translate(req: TranslateRequest):
    try:
        print(f"\n=== Translation Request ===")
        print(f"Model: {req.model}")
        print(f"Text: {req.text}")
        
        # Хэлний кодыг тохируулах
        src_lang, tgt_lang = get_lang_codes_for_model(req.model, req.src_lang, req.tgt_lang)
        print(f"Source: {req.src_lang} -> {src_lang}")
        print(f"Target: {req.tgt_lang} -> {tgt_lang}")
        
        # Шинэ endpoint-тэй InferenceClient ашиглах
        client = get_inference_client()
        
        # Орчуулга хийх
        try:
            # Тусгайлан NLLB model-д зориулсан арга
            if "nllb" in req.model.lower():
                print(f"Using NLLB model with src={src_lang}, tgt={tgt_lang}")
                
                # NLLB model-д зориулсан параметрүүд
                result = client.post(
                    json={
                        "inputs": req.text,
                        "parameters": {
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang
                        }
                    },
                    model=req.model
                )
                
                # JSON хариултыг задлах
                if isinstance(result, bytes):
                    result = json.loads(result.decode('utf-8'))
                
                print(f"NLLB raw result: {result}")
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "generated_text" in result[0]:
                        translation = result[0]["generated_text"]
                    elif isinstance(result[0], dict) and "translation_text" in result[0]:
                        translation = result[0]["translation_text"]
                    else:
                        translation = str(result[0])
                elif isinstance(result, dict) and "generated_text" in result:
                    translation = result["generated_text"]
                elif isinstance(result, dict) and "translation_text" in result:
                    translation = result["translation_text"]
                else:
                    translation = str(result)
                    
            elif "t5" in req.model.lower() or "mt5" in req.model.lower():
                # T5 model-уудад prompt нэмэх
                prompt = f"translate {src_lang} to {tgt_lang}: {req.text}"
                print(f"T5 Prompt: {prompt}")
                result = client.translation(prompt, model=req.model)
                
                if hasattr(result, 'translation_text'):
                    translation = result.translation_text
                elif isinstance(result, str):
                    translation = result
                else:
                    translation = str(result)
                    
            else:
                # Бусад model-ууд (MBART, M2M100)
                print(f"Using translation API with: src={src_lang}, tgt={tgt_lang}")
                result = client.translation(
                    req.text,
                    model=req.model,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang
                )
                
                if hasattr(result, 'translation_text'):
                    translation = result.translation_text
                elif isinstance(result, list) and len(result) > 0:
                    translation = result[0].translation_text
                else:
                    translation = str(result)
            
        except Exception as api_error:
            print(f"Primary API method failed: {str(api_error)}")
            # Өөр аргыг турших
            raise api_error
        
        translation = translation.strip()
        print(f"Translation: {translation}")
        
        if not translation or translation == "":
            raise ValueError("Хоосон орчуулга")
        
        return {
            "success": True,
            "chatInput": req.text,
            "chatInputEn": translation,
            "model": req.model,
            "src_lang_used": src_lang,
            "tgt_lang_used": tgt_lang,
            "note": "Success with original model"
        }
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error: {error_msg}")
        print(f"Trace: {error_trace}")
        
        # Fallback: MBART model
        try:
            print("Trying fallback to MBART...")
            client = get_inference_client()
            
            # MBART model-д зориулсан шууд дуудлага
            result = client.post(
                json={
                    "inputs": req.text,
                    "parameters": {
                        "src_lang": "mn_MN",
                        "tgt_lang": "en_XX"
                    }
                },
                model="facebook/mbart-large-50-many-to-many-mmt"
            )
            
            if isinstance(result, bytes):
                result = json.loads(result.decode('utf-8'))
            
            print(f"MBART fallback raw: {result}")
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "translation_text" in result[0]:
                    translation = result[0]["translation_text"]
                else:
                    translation = str(result[0])
            elif isinstance(result, dict) and "translation_text" in result:
                translation = result["translation_text"]
            else:
                translation = str(result)
            
            translation = translation.strip()
            
            return {
                "success": True,
                "chatInput": req.text,
                "chatInputEn": translation,
                "model": "facebook/mbart-large-50-many-to-many-mmt",
                "src_lang_used": "mn_MN",
                "tgt_lang_used": "en_XX",
                "note": f"Fallback used due to error with {req.model}"
            }
            
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return {
                "success": False,
                "error": f"Main error: {error_msg}",
                "fallback_error": str(fallback_error)
            }

@app.post("/translate/nllb-simple")
async def translate_nllb_simple(text: str):
    """NLLB model-д зориулсан энгийн endpoint"""
    try:
        client = get_inference_client()
        
        result = client.post(
            json={
                "inputs": text,
                "parameters": {
                    "src_lang": "mon_Cyrl",
                    "tgt_lang": "eng_Latn"
                }
            },
            model="facebook/nllb-200-distilled-600M"
        )
        
        if isinstance(result, bytes):
            result = json.loads(result.decode('utf-8'))
        
        print(f"NLLB simple raw: {result}")
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                translation = result[0]["generated_text"]
            else:
                translation = str(result[0])
        else:
            translation = str(result)
        
        return {
            "success": True,
            "original": text,
            "translation": translation.strip(),
            "model": "facebook/nllb-200-distilled-600M"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/translate/mbart-simple")
async def translate_mbart_simple(text: str):
    """MBART model-д зориулсан энгийн endpoint"""
    try:
        client = get_inference_client()
        
        result = client.post(
            json={
                "inputs": text,
                "parameters": {
                    "src_lang": "mn_MN",
                    "tgt_lang": "en_XX"
                }
            },
            model="facebook/mbart-large-50-many-to-many-mmt"
        )
        
        if isinstance(result, bytes):
            result = json.loads(result.decode('utf-8'))
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "translation_text" in result[0]:
                translation = result[0]["translation_text"]
            else:
                translation = str(result[0])
        else:
            translation = str(result)
        
        return {
            "success": True,
            "original": text,
            "translation": translation.strip(),
            "model": "facebook/mbart-large-50-many-to-many-mmt"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test/all-models")
async def test_all_models():
    """Бүх model-уудыг турших"""
    test_text = "Сайн байна уу?"
    
    test_cases = [
        {
            "name": "MBART",
            "model": "facebook/mbart-large-50-many-to-many-mmt",
            "src": "mn_MN",
            "tgt": "en_XX",
            "method": "translation"
        },
        {
            "name": "NLLB",
            "model": "facebook/nllb-200-distilled-600M",
            "src": "mon_Cyrl",
            "tgt": "eng_Latn",
            "method": "post"
        },
        {
            "name": "M2M100",
            "model": "facebook/m2m100_418M",
            "src": "mn",
            "tgt": "en",
            "method": "translation"
        }
    ]
    
    results = []
    for test in test_cases:
        try:
            client = get_inference_client()
            
            if test["method"] == "post":
                result = client.post(
                    json={
                        "inputs": test_text,
                        "parameters": {
                            "src_lang": test["src"],
                            "tgt_lang": test["tgt"]
                        }
                    },
                    model=test["model"]
                )
                
                if isinstance(result, bytes):
                    result = json.loads(result.decode('utf-8'))
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        translation = result[0].get("generated_text", 
                                                   result[0].get("translation_text", 
                                                                str(result[0])))
                    else:
                        translation = str(result[0])
                else:
                    translation = str(result)
                    
            else:  # translation method
                result = client.translation(
                    test_text,
                    model=test["model"],
                    src_lang=test["src"],
                    tgt_lang=test["tgt"]
                )
                
                if hasattr(result, 'translation_text'):
                    translation = result.translation_text
                elif isinstance(result, list) and len(result) > 0:
                    translation = result[0].translation_text
                else:
                    translation = str(result)
            
            results.append({
                "model": test["name"],
                "id": test["model"],
                "status": "success",
                "translation": translation.strip()
            })
            
        except Exception as e:
            results.append({
                "model": test["name"],
                "id": test["model"],
                "status": "error",
                "error": str(e)[:200]
            })
    
    return {"test_results": results}

@app.get("/")
async def root():
    {
        "status": "Ready (no featured, direct translation)"
        , "usage": "POST /translate body-д 'model' солих боломжтой"
        , "recommended_models_mn_en": [
            { "model"   : "facebook/mbart-large-50-many-to-many-mmt"    , "desc": "Default, stable for mn_MN → en_XX"}      # Ажиллана
            , { "model" : "facebook/m2m100_418M"                        , "desc": "Fast, mn → en (100 languages)"}          # 
            , { "model" : "facebook/nllb-200-distilled-600M"            , "desc": "200 languages, mon_Cyrl → eng_Latn"}     # 
            , { "model" : "google/mt5-base"                             , "desc": "High quality, auto-prompt added"}        # 
        ]
    }
    return {
        "status": "Translation API (Updated for new HuggingFace endpoint)",
        "endpoints": {
            "POST /translate": "Ерөнхий орчуулга (JSON body)",
            "POST /translate/nllb-simple?text=...": "Шууд NLLB орчуулга",
            "POST /translate/mbart-simple?text=...": "Шууд MBART орчуулга",
            "GET /test/all-models": "Бүх model-уудыг турших"
        },
        "api_info": {
            "base_url": "https://router.huggingface.co/huggingface",
            "note": "Updated for new HuggingFace API endpoint"
        },
        "example_curl": [
            'curl -X POST http://localhost:8000/translate/nllb-simple?text="Сайн+байна+уу"',
            'curl -X POST http://localhost:8000/translate -H "Content-Type: application/json" -d \'{"text":"Сайн байна уу", "model":"facebook/nllb-200-distilled-600M"}\''
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)