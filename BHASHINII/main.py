from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer
import faiss
import torch
import pandas as pd
import speech_recognition as sr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ✅ Define FastAPI app
app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load English ↔ Hindi Translation models
en_hi_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
en_hi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
hi_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
hi_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")

# ✅ Load FAISS index (hindi_english_mappings.csv)
faiss_index = faiss.read_index("C:\\Users\\SAI CHARAN RAJU\\OneDrive\\Desktop\\BHASHINII\\hindi_english.index")  # Ensure you have the .index file

# ✅ Load the dataset (scraped_hindi_data.csv)
df = pd.read_csv("C:\\Users\\SAI CHARAN RAJU\\OneDrive\\Desktop\\BHASHINII\\scraped_hindi_data (1).csv")

# ✅ Translation APIs
@app.get("/translate/en-hi")
async def translate_en_hi(text: str):
    try:
        with torch.no_grad():
            inputs = en_hi_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = en_hi_model.generate(**inputs)
            translation = en_hi_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return {"english": text, "hindi": translation[0]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/translate/hi-en")
async def translate_hi_en(text: str):
    try:
        with torch.no_grad():
            inputs = hi_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = hi_en_model.generate(**inputs)
            translation = hi_en_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return {"hindi": text, "english": translation[0]}
    except Exception as e:
        return {"error": str(e)}

# ✅ FAISS Search API (Find similar translations)
@app.get("/faiss/search")
async def faiss_search(query: str, k: int = 5):
    try:
        # Convert query to vector (you can use a model to embed the query)
        query_embedding = en_hi_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_vector = en_hi_model.encode(**query_embedding).detach().numpy()

        # Perform FAISS search
        distances, indices = faiss_index.search(query_vector, k)
        
        # Retrieve the corresponding translations
        results = []
        for index in indices[0]:
            result = df.iloc[index]["hindi_translation"]  # Change column name based on your CSV structure
            results.append(result)
        
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}

# ✅ Speech-to-Text API (Supports Both Hindi & English)
@app.get("/speech-to-text")
async def speech_to_text(lang: str = "en-US"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)  # Add timeout to prevent infinite listening
            text = recognizer.recognize_google(audio, language=lang)
            return {"text": text}
        except sr.UnknownValueError:
            return {"error": "Could not understand the speech."}
        except sr.RequestError:
            return {"error": "Speech recognition service is unavailable."}
        except Exception as e:
            return {"error": str(e)}

# ✅ Run the FastAPI server using:
# uvicorn main:app --reload
