import os
from transformers import AutoModel, AutoTokenizer

# Initialisiere das OCR-Modell
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0',
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().cuda()

# Pfade zur Eingabedatei und Ausgabedatei
input_image_path = '../Bilder/Kopf_Fußzeilen/Kopf_Fußzeilen_7.png'  # Pfad zur Bilddatei
output_folder = '../Modell_Output/GotOcr/Kopf_Fußzeilen'  # Zielordner
output_file = os.path.join(output_folder, 'Kopf_Fußzeilen_7_output.txt')  # Zieldatei für den Text

# Stelle sicher, dass das Ausgabeverzeichnis existiert
os.makedirs(output_folder, exist_ok=True)

# Führe OCR auf dem Bild durch
text = model.chat(tokenizer, input_image_path, ocr_type='ocr')

# Speichere den extrahierten Text in einer Textdatei
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(text)

print(f'Text für {input_image_path} wurde gespeichert in {output_file}')
