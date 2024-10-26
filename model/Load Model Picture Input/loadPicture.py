import torch
from PIL import Image
from transformers import Kosmos2_5ForConditionalGeneration, Kosmos2_5Processor

# Lade den Processor und das Modell
processor = Kosmos2_5Processor.from_pretrained("microsoft/kosmos-2.5")
model = Kosmos2_5ForConditionalGeneration.from_pretrained("microsoft/kosmos-2.5")

# Lade das Bild (du kannst das Bild vorher in JPG konvertieren, wenn nötig)
image_path = "testbilder/receipt_00008.jpg"  # Pfad zu deinem Bild
image = Image.open(image_path)

# Verarbeite das Bild
inputs = processor(images=image, return_tensors="pt")

# Extrahiere die wichtigen Informationen
flattened_patches = inputs["flattened_patches"]
attention_mask = inputs["attention_mask"]

# Setze das Modell in den Evaluierungsmodus
model.eval()

# Generiere Text basierend auf den Bilddaten, erhöhe die maximale Anzahl der Tokens
with torch.no_grad():
    outputs = model.generate(flattened_patches=flattened_patches, 
                             attention_mask=attention_mask, 
                             max_new_tokens=100)  # Erhöhe die Token-Anzahl

# Decodiere die generierten Token in Text
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Ausgabe des extrahierten Textes
print("Erkannter Text:", generated_text)
