import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import numpy as np

# Initialisations globales
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def extract_text_from_pdf(path: str) -> str:
    """
    Extrait le texte d’un PDF page par page.
    """
    try:
        reader = PyPDF2.PdfReader(path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        return full_text
    except Exception as e:
        print("Erreur lors de l’extraction PDF :", e)
        return ""

def chunk_text(text: str, max_length: int = 1500) -> list:
    """
    Divise le texte en morceaux de taille donnée.
    """
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def build_faiss_index(chunks: list) -> tuple:
    """
    Crée l’index FAISS à partir des embeddings de chunks.
    """
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, chunks

def query_pdf(question: str, index, chunks: list, embeddings) -> str:
    """
    À partir d’une question, récupère les passages pertinents et génère une réponse.
    """
    question_embedding = embedder.encode([question])
    _, top_indices = index.search(np.array(question_embedding), k=3)
    top_chunks = [chunks[i] for i in top_indices[0]]

    context = "\n".join(top_chunks)
    prompt = f"Question: {question}\nContexte: {context}\nRéponse:"
    response = qa_pipeline(prompt, max_length=100)[0]["generated_text"]
    return response
