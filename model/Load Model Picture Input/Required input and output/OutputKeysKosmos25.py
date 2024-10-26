import torch
from PIL import Image
from transformers import Kosmos2_5ForConditionalGeneration, Kosmos2_5Processor

# Lade den Tokenizer/Processor
processor = Kosmos2_5Processor.from_pretrained("microsoft/kosmos-2.5")
model = Kosmos2_5ForConditionalGeneration.from_pretrained("microsoft/kosmos-2.5")

# Lade ein Bild
image_path = "../testbilder/in.png"  # Gib den Pfad zu deinem Bild an
image = Image.open(image_path)

# Verarbeite das Bild mit dem Processor und gebe die Tensoren zurück
inputs = processor(images=image, return_tensors="pt")

# Überprüfe die verfügbaren Schlüssel in 'inputs'
print("Verfügbare Schlüssel in 'inputs':", inputs.keys())

# Setze das Modell in den Evaluierungsmodus
model.eval()



#Codebeschreibung:
#1. Kosmos2_5Processor verarbeitet bild
#2. Überprüft Schlüssel die im input-Dictionary nach der VErarbeitung vorhanden sind -> Welche Schlüssel verfügbar sind

#Ergebnis:
#--> Kosmos gibt die Schlüssel flattened_patches, attention_mask, width und height zurück
# flatterned_patches die Bilddaten in einer Form repräsentiert, die vom Modell verarbeitet werden kann wieder