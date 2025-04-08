from modules.text_analyzer import analyze_text
from modules.pdf_analyzer import extract_text_from_pdf, chunk_text, build_faiss_index, query_pdf
from modules.image_analyzer import analyze_image

def test_text():
    print("TEST - Analyse de texte")
    texte = """
    Artificial Intelligence (AI) is transforming the way we live and work. From healthcare to transportation, AI-powered systems are enabling faster decision-making, improving efficiency, and opening up new possibilities. In the field of healthcare, for example, AI algorithms are used to detect diseases such as cancer at earlier stages, sometimes with higher accuracy than human doctors.
    In finance, AI is used for fraud detection, algorithmic trading, and personalized banking experiences. Meanwhile, autonomous vehicles rely on AI to interpret sensor data and make real-time driving decisions. These innovations are not without challenges, including ethical concerns, job displacement, and the need for robust data privacy protections.
    Despite these hurdles, the growth of AI continues at a rapid pace, driven by advances in deep learning, increased computational power, and the availability of massive datasets. Companies and governments around the world are investing heavily in AI research and development, aiming to maintain competitive advantages and address pressing societal issues.
    As AI becomes more integrated into our lives, it is crucial to ensure that these technologies are developed and deployed responsibly. This includes addressing biases in algorithms, maintaining transparency, and creating regulations that balance innovation with public trust.
    """
    résumé = analyze_text(texte)
    print("Résumé :", résumé)

def test_pdf():
    print("\n TEST - Analyse de PDF")
    chemin_pdf = "assets/sample.pdf"  # Remplace par le chemin réel
    texte = extract_text_from_pdf(chemin_pdf)
    
    if not texte.strip():
        print(" PDF vide ou non lisible.")
        return

    chunks = chunk_text(texte)
    index, embeddings, chunks = build_faiss_index(chunks)
    
    question = "Quel est le thème principal du document ?"
    réponse = query_pdf(question, index, chunks, embeddings)
    print("Réponse :", réponse)

def test_image():
    print("\n TEST - Analyse d'image")
    chemin_image = "assets/image.jpg"  # Remplace par le chemin réel
    résultat = analyze_image(chemin_image)
    print(résultat)

if __name__ == "__main__":
    print("===== TEST GLOBAL DU PROJET LLM =====\n")
    
    test_text()
    test_pdf()
    test_image()
    
    print("\n Tous les tests sont terminés.")
