from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

app = FastAPI()
client = InferenceClient(token=os.getenv("HF_TOKEN"))

class TranslateRequest(BaseModel):
    text       : str
    src_lang   : str = "mn_MN"      # MBART/M2M100/NLLB-д
    tgt_lang   : str = "en_XX"      # MBART/M2M100/NLLB-д
    model      : str = "facebook/mbart-large-50-many-to-many-mmt"  # Default (тогтвортой)

@app.post("/translate")
async def translate(req: TranslateRequest):
    try:
        print(f"Request → Model: {req.model} | {req.src_lang} → {req.tgt_lang} | Text: {req.text[:60]}")

        # === 1. mT5 / T5 серийн онцлог (prompt шаардлагатай, src/tgt_lang дэмждэггүй) ===
        if "t5" in req.model.lower() or "mt5" in req.model.lower():
            # Prompt автоматаар нэмэх (2025 syntax-д баталгаатай)
            prompt_text = f"translate Mongolian to English: {req.text}"
            result = client.translation(
                prompt_text,
                model=req.model
            )
            translation = (
                result.translation_text if hasattr(result, "translation_text")
                else result[0].translation_text if isinstance(result, list) else str(result)
            ).strip()
        else:
            # === 2. Бусад модель (MBART, NLLB, M2M100 гэх мэт) ===
            result = client.translation(
                req.text,
                model=req.model,
                src_lang=req.src_lang,
                tgt_lang=req.tgt_lang
            )
            translation = (
                result.translation_text if hasattr(result, "translation_text")
                else result[0].translation_text if isinstance(result, list) else str(result)
            ).strip()

        if not translation:
            raise ValueError("Empty translation result")

        print(f"Success → {translation}")
        return {"success": True, "chatInput": req.text, "chatInputEn": translation}

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"Error: {error_msg}\n{error_trace}")
        
        # Fallback: Default MBART руу шилжих (provider алдаанаас сэргийлэх)
        try:
            print("Falling back to default MBART model...")
            fallback_result = client.translation(
                req.text,
                model="facebook/mbart-large-50-many-to-many-mmt",
                src_lang="mn_MN",
                tgt_lang="en_XX"
            )
            fallback_translation = (
                fallback_result.translation_text if hasattr(fallback_result, "translation_text")
                else fallback_result[0].translation_text
            ).strip()
            return {
                "success": True
                , "chatInput": req.text
                , "chatInputEn": fallback_translation
                , "model": "facebook/mbart-large-50-many-to-many-mmt"
                , "note": "Used fallback model due to error"
            }
        except:
            return {"success": False, "error": error_msg, "details": error_trace[:400], "fallback_failed": True}

@app.get("/")
async def root():
    return {
        "status": "Ready (no featured, direct translation)",
        "usage": "POST /translate body-д 'model' солих боломжтой",
        "recommended_models_mn_en": [
            {"model": "facebook/mbart-large-50-many-to-many-mmt", "desc": "Default, stable for mn_MN → en_XX"},
            {"model": "facebook/m2m100_418M", "desc": "Fast, mn → en (100 languages)"},
            {"model": "facebook/nllb-200-distilled-600M", "desc": "200 languages, mon_Cyrl → eng_Latn"},
            {"model": "google/mt5-base", "desc": "High quality, auto-prompt added"}
        ]
    }