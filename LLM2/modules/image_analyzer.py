from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import torch
import os

# Chargement du modèle et du processor BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Chargement du pipeline de résumé
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def describe_image(image_path: str) -> str:
    """
    Génère une description (caption) de l’image.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    except Exception as e:
        print(f"Erreur lors de la description de l’image : {e}")
        return ""

def summarize_caption(caption: str) -> str:
    """
    Résume la caption générée (optionnel si caption longue).
    """
    if len(caption) < 100:
        return caption  # inutile de résumer une phrase courte
    summary = summarizer(caption, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

def analyze_image(image_path: str) -> str:
    """
    Analyse complète de l’image : caption + résumé.
    """
    caption = describe_image(image_path)
    if not caption:
        return "Impossible d’analyser l’image."

    summary = summarize_caption(caption)
    return f"Résumé : {summary}"
