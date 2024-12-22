from PIL import Image
import pytesseract
from langchain_ollama import OllamaLLM

# Configurez Tesseract (ajustez le chemin si nécessaire)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chemin de l'image
image_path = r"C:\Users\MSI\Downloads\tech.png"


# Extraire le texte de l'image
image = Image.open(image_path)
extracted_text = pytesseract.image_to_string(image)

# Utiliser le modèle LLaVA pour analyser le texte extrait
llm = OllamaLLM(model="llava")
prompt = f"Voici un texte extrait d'une image : {extracted_text}. Quelles sont les compétences techniques clés mentionnées ?"

response = llm.invoke(prompt)

print("Compétences clés extraites :")
print(response)
