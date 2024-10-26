import torch
from PIL import Image
from transformers import Kosmos2_5ForConditionalGeneration, AutoProcessor

# Konfiguration
repo = "microsoft/kosmos-2.5"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Falls du eine GPU hast, benutze sie
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # Optimiert für GPU

# Lade das Modell und den Prozessor (ohne device_map)
model = Kosmos2_5ForConditionalGeneration.from_pretrained(repo, torch_dtype=dtype)
model.to(device)  # Manuelle Übertragung auf die GPU/CPU

processor = AutoProcessor.from_pretrained(repo)

# Lade dein Bild
image_path = "testbilder/oldNewspaper.jpg"
image = Image.open(image_path)

# Verwende ein Text-Prompt, wenn nötig
prompt = "<md>"

# Verarbeite Bild und Text mit dem Prozessor
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
        max_new_tokens=1024  # Erhöhe die maximale Token-Anzahl für längere Textausgaben
    )

# Decodiere die generierten Token in Text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("Erkannter Text:", generated_text[0])

#BEschreibung
# flattened_patches enthält wahrscheinlich die Bilddaten, die das Modell erwartet
# -> Übergabe der Schlüssel an das Model
# attention_mask enthält Informationen welcher Teil des Bildes relevant ist

