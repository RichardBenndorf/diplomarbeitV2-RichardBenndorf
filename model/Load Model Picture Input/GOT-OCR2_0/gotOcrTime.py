import os
import time
from transformers import AutoModel, AutoTokenizer

# Initialisiere das OCR-Modell
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# Pfad zum Verzeichnis mit den Testbildern
input_folder = '../Bilder/Tabellenformat'
output_folder = '../Modell_Output/GotOcr/Tabellenformat'
os.makedirs(output_folder, exist_ok=True)

# Datei für Durchlaufzeiten vorbereiten
durations_path = os.path.join(output_folder, "durations.txt")
with open(durations_path, 'w', encoding='utf-8') as f:
    f.write("Modell Ladezeit, Durchlaufzeit (Sekunden)\n")

# Dummy-Bild laden, um das Modell zu initialisieren
dummy_image_path = '../DummyPicture/dummy.png'  # Pfad zur Dummy-Datei
start_time = time.time()
model.chat(tokenizer, dummy_image_path, ocr_type='format')  # Dummy-Durchlauf zur Initialisierung
model_load_duration = time.time() - start_time

# Modell Ladezeit in der Datei speichern
with open(durations_path, 'a', encoding='utf-8') as f:
    f.write(f"Modell Ladezeit, {model_load_duration:.2f} Sekunden\n")

# Durchlaufe alle Bilder im Verzeichnis
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):  # Nur PNG-Bilder verarbeiten
        image_path = os.path.join(input_folder, filename)
        
        # Startzeit für die Bildverarbeitung messen
        start_time = time.time()
        
        # Führe OCR auf dem Bild durch
        text = model.chat(tokenizer, image_path, ocr_type='ocr')
        processing_time = time.time() - start_time  # Berechne die Durchlaufzeit
        
        # Pfad für die Ausgabedatei
        output_path = os.path.join(output_folder, filename.replace('.png', '_output.txt'))
        
        # Speichere den extrahierten Text in einer Textdatei
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        
        # Speichere die Durchlaufzeit in der Datei `durations.txt`
        with open(durations_path, 'a', encoding='utf-8') as f:
            f.write(f"{filename}, {processing_time:.2f}\n")
        
        print(f'Text für {filename} wurde gespeichert in {output_path} mit einer Durchlaufzeit von {processing_time:.2f} Sekunden.')
