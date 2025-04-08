import streamlit as st
from modules.text_analyzer import analyze_text
from modules.pdf_analyzer import extract_text_from_pdf, chunk_text, build_faiss_index, query_pdf
from modules.image_analyzer import analyze_image
import tempfile
import os

st.set_page_config(page_title="LLM Content Analyzer", layout="centered")

st.title("LLM Content Analyzer")
st.markdown("Analysez du texte, des documents PDF ou des images avec des modèles LLM.")

# Navigation
mode = st.sidebar.radio("Choisissez le type d'analyse :", ("Texte", "PDF", "Image"))

# --------------------
# Texte
# --------------------
if mode == "Texte":
    st.subheader("Analyse de Texte")
    user_input = st.text_area("Entrez votre texte ici :", height=200)

    if st.button("Analyser le texte"):
        if user_input.strip():
            summary = analyze_text(user_input)
            st.success("Résumé généré :")
            st.write(summary)
        else:
            st.warning("Veuillez entrer un texte à analyser.")

# --------------------
# PDF
# --------------------
elif mode == "PDF":
    st.subheader("Analyse de PDF avec Question")
    uploaded_pdf = st.file_uploader("Chargez un fichier PDF", type="pdf")
    user_question = st.text_input("Posez une question sur le document :")

    if uploaded_pdf and user_question:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name

        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text, max_length=500)
        index, embeddings, chunks = build_faiss_index(chunks)
        response = query_pdf(user_question, index, chunks, embeddings)
        os.remove(tmp_path)

        st.success("Réponse générée :")
        st.write(response)

# --------------------
# Image
# --------------------
elif mode == "Image":
    st.subheader("Analyse d’image")
    uploaded_image = st.file_uploader("Chargez une image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_image.read())
            tmp_img_path = tmp_img.name

        st.image(uploaded_image, caption="Image chargée", use_column_width=True)

        result = analyze_image(tmp_img_path)
        st.success("Analyse de l'image :")
        st.write(result)

        os.remove(tmp_img_path)
