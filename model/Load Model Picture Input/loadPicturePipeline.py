import torch
from PIL import Image
from transformers import Kosmos2_5ForConditionalGeneration, AutoProcessor
import os

# Konfiguration
repo = "microsoft/kosmos-2.5"
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU verwenden, falls verfügbar
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # Optimierung für GPU

# Lade Modell und Prozessor
model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype)
model.to(device)

processor = AutoProcessor.from_pretrained(repo)

# Verzeichnispfade
image_folder = "Bilder/Tabellenformat"
output_folder = "Modell_Output/Kosmos/Tabellenformat"
os.makedirs(output_folder, exist_ok=True)

# Schleife über alle Bilder im Verzeichnis
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        # Textprompt
        prompt = "<md>"
    
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

        # Decodiere die generierten Token in Text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Speichere die Modellausgabe in einer Textdatei
        output_file = os.path.join(output_folder, f"{filename.split('.')[0]}_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)

        print(f"Text aus {filename} in {output_file} gespeichert.")
