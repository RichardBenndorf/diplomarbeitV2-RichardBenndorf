import torch
from PIL import Image
from transformers import Kosmos2_5ForConditionalGeneration, AutoProcessor
import os
import time

# Konfiguration
repo = "microsoft/kosmos-2.5"
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU verwenden, falls verfügbar
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # Optimierung für GPU

# Lade Modell und Prozessor
model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype)
model.to(device)

processor = AutoProcessor.from_pretrained(repo)

# Verzeichnispfade
image_folder = "Bilder/uneinheitliches Layout"
output_folder = "Modell_Output/Kosmos/uneinheitliches Layout"
os.makedirs(output_folder, exist_ok=True)

# Datei für Durchlaufzeiten vorbereiten
durations_path = os.path.join(output_folder, "durations.txt")
with open(durations_path, 'w', encoding='utf-8') as f:
    f.write("Dateiname, Durchlaufzeit (Sekunden)\n")

# Schleife über alle Bilder im Verzeichnis
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        # Textprompt
        prompt = "<md>"

        # Startzeit für die Verarbeitung messen
        start_time = time.time()
        
        # Verarbeite das Bild
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        # Entferne 'width' und 'height' aus den Eingaben
        inputs.pop("width", None)
        inputs.pop("height", None)

        # Übertrage die Eingaben auf das richtige Gerät (CPU oder GPU)
        inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
        inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)

        # Generiere Text basierend auf dem Bild
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024  # Maximale Anzahl der Tokens erhöhen für lange Textausgabe
            )

        # Berechne die Durchlaufzeit
        processing_time = time.time() - start_time
        
        # Decodiere die generierten Token in Text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Speichere die Modellausgabe in einer Textdatei
        output_file = os.path.join(output_folder, f"{filename.split('.')[0]}_output.txt")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(generated_text)

        # Speichere die Durchlaufzeit in der Datei `durations.txt`
        with open(durations_path, 'a', encoding='utf-8') as f:
            f.write(f"{filename}, {processing_time:.2f}\n")

        print(f"Text aus {filename} in {output_file} gespeichert. Durchlaufzeit: {processing_time:.2f} Sekunden.")
