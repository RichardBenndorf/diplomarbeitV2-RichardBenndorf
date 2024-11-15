import os
from transformers import AutoModel, AutoTokenizer

# Initialisiere das OCR-Modell
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# Pfad zum Verzeichnis mit den Testbildern
input_folder = '../Bilder/uneinheitliches Layout'
output_folder = '../Modell_Output/GotOcr/uneinheitliches Layout'

# Stelle sicher, dass das Ausgabeverzeichnis existiert
os.makedirs(output_folder, exist_ok=True)

# Durchlaufe alle Bilder im Verzeichnis
for filename in os.listdir(input_folder):
    # Stelle sicher, dass nur Bilddateien verarbeitet werden (optional kannst du nach spezifischen Dateiendungen filtern)
    if filename.endswith('.png'):  # Beispielweise können hier andere Formate wie '.jpg' ebenfalls berücksichtigt werden
        image_path = os.path.join(input_folder, filename)
        # Führe OCR auf dem Bild durch
        text = model.chat(tokenizer, image_path, ocr_type='format')
        
        # Speichere den extrahierten Text in einer Textdatei
        output_path = os.path.join(output_folder, filename.replace('.png', '.txt'))
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)

        print(f'Text für {filename} wurde gespeichert in {output_path}')
