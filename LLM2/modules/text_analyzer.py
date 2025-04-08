from transformers import pipeline, Pipeline
from typing import Optional

# Chargement du modèle BART pour le résumé
try:
    summarizer: Pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print("Erreur lors du chargement du modèle de summarization :", e)
    summarizer = None

def analyze_text(text: str, chunk_size: int = 1000) -> Optional[str]:
    """
    Résume un texte en utilisant le modèle de summarization Hugging Face.

    Args:
        text (str): Le texte à résumer.
        chunk_size (int): Longueur maximale d'un segment (limite du modèle).

    Returns:
        Optional[str]: Le résumé généré, ou None en cas d’erreur.
    """
    if not text.strip():
        print("Texte vide.")
        return None

    if summarizer is None:
        print("Modèle non chargé.")
        return None

    try:
        summaries = []
        # Découpage du texte si trop long
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])

        return "\n\n".join(summaries)

    except Exception as e:
        print("Erreur pendant l’analyse du texte :", e)
        return None

