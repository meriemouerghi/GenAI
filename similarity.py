from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

llm = OllamaLLM(model="llama3.2")

def extract_key_sentences(text):
    prompt = f"Extrayez les technical and soft skills \n\n{text}\n\nPhrases clés :"
    response = llm.invoke(prompt)
    return response

def calculate_similarity(job_skills, cv_skills):
    texts = [job_skills, cv_skills]
    vectorizer = CountVectorizer().fit_transform(texts)
    cosine_sim = cosine_similarity(vectorizer)
    return cosine_sim[0][1]

# Chemins des fichiers PDF
job_pdf_path = r"C:\Users\MSI\Downloads\job_offer.pdf"
cv_pdf_path = r"C:\Users\MSI\Downloads\CV-ArwaGaied.pdf"  # Remplacez par le chemin réel du CV

# Extraction du texte des PDFs
job_pdf_text = extract_text_from_pdf(job_pdf_path)
cv_pdf_text = extract_text_from_pdf(cv_pdf_path)

# Extraction des compétences
job_skills = extract_key_sentences(job_pdf_text)
cv_skills = extract_key_sentences(cv_pdf_text)

# Calcul de la similarité
similarity_score = calculate_similarity(job_skills, cv_skills)

print("Compétences extraites de l'offre d'emploi :")
print(job_skills)
print("\nCompétences extraites du CV :")
print(cv_skills)
print(f"\nScore de similarité : {similarity_score}")
