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

# Datei- und Ausgabepfade
image_path = "Bilder/Tabellenformat/tabellenformat_8.png"  # Pfad zum Bild
output_folder = "Modell_Output/Kosmos/Tabellenformat"  # Zielverzeichnis
output_file = os.path.join(output_folder, "tabellenformat_8_output.txt")  # Ziel-Datei für die Textausgabe
durations_path = os.path.join(output_folder, "durations2.txt")  # Datei für Durchlaufzeit

# Sicherstellen, dass das Zielverzeichnis existiert
os.makedirs(output_folder, exist_ok=True)

# Textprompt
prompt = "<md>"

# Startzeit für die Verarbeitung messen
start_time = time.time()

# Bild laden
image = Image.open(image_path)

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
        max_new_tokens=2048  # Maximale Anzahl der Tokens erhöhen für lange Textausgabe
    )

# Berechne die Durchlaufzeit
processing_time = time.time() - start_time

# Decodiere die generierten Token in Text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Speichere die Modellausgabe in einer Textdatei
with open(output_file, "w", encoding='utf-8') as f:
    f.write(generated_text)

# Speichere die Durchlaufzeit in der Datei `durations.txt`
with open(durations_path, 'w', encoding='utf-8') as f:
    f.write("Dateiname, Durchlaufzeit (Sekunden)\n")
    f.write(f"{os.path.basename(image_path)}, {processing_time:.2f}\n")

# Erfolgsmeldung ausgeben
print(f"Text aus {image_path} in {output_file} gespeichert. Durchlaufzeit: {processing_time:.2f} Sekunden.")
