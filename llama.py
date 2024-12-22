from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")

def extract_key_sentences(text):
    prompt = f"Extrayez les technical and soft skills \n\n{text}\n\nPhrases clés :"
    response = llm(prompt)
    return response

pdf_path = r"C:\Users\MSI\Desktop\UIK\JD\Job Descriptions\CONSULTING\TECHNO\JD Consultant SAS Sénior.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
key_sentences = extract_key_sentences(pdf_text)

print("Phrases clés extraites :")
print(key_sentences)
