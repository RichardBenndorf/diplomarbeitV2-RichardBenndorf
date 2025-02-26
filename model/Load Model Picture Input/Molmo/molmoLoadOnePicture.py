import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os

# Modell und Prozessor laden
processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto')
model = AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto')

# Pfad zum Bild und zum Output-Ordner festlegen
image_filename = 'Kopf_Fußzeilen_7.png'
images_directory = '../Bilder/Kopf_Fußzeilen'
output_directory = '../Modell_Output/Molmo/Kopf_Fußzeilen'

# Output-Ordner erstellen, falls er nicht existiert
os.makedirs(output_directory, exist_ok=True)

# Pfad zur Bilddatei
image_path = os.path.join(images_directory, image_filename)

# Überprüfen, ob es sich um eine Bilddatei handelt
if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    image = Image.open(image_path)

    # Bild verarbeiten mit spezifischer Aufforderung
    inputs = processor.process(images=[image], text="Please extract the text from the image and don't add any aditional words.")
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=1000, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # Ergebnis ausgeben
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Text in einer Datei speichern, Dateiname wie das Bild
    output_text_path = os.path.join(output_directory, os.path.splitext(image_filename)[0] + '.txt')
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(generated_text)
    print(f"Text for {image_filename} saved to {output_text_path}")
