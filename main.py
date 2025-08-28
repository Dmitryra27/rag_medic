# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import os

# === Инициализация ===
app = FastAPI()

# Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("medical_knowledge")

# Модели
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@004")
gemini_model = GenerativeModel("gemini-1.5-flash")

# === Модель запроса ===
class QuestionRequest(BaseModel):
    question: str

# === Поиск по базе знаний ===
def search_knowledge(query, n=3):
    query_embedding = embedding_model.get_embeddings([query])[0].values
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )
    return results["documents"][0], results["metadatas"][0]

# === Генерация ответа ===
@app.post("/ask")
def ask(request: QuestionRequest):
    try:
        contexts, sources = search_knowledge(request.question)
        context_text = "\n\n".join(contexts)

        prompt = f"""
        Ты — медицинский ассистент. Отвечай на вопрос, опираясь только на предоставленный контекст.
        Если ответа нет в контексте, скажи: "Я не могу дать медицинскую консультацию. Обратитесь к врачу."

        Контекст:
        {context_text}

        Вопрос: {request.question}
        Ответ:
        """

        response = gemini_model.generate_content(prompt)
        answer = response.text

        return {
            "question": request.question,
            "answer": answer,
            "sources": [s["source"] for s in sources]
        }
    except Exception as e:
        return {"error": str(e)}

# === Тестовый эндпоинт ===
@app.get("/")
def home():
    return {"status": "ok", "message": "RAG-сервер запущен"}
