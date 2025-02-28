import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

# Modell und Prozessor laden
processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto')
model = AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto')

# Pfade für Eingabe und Ausgabe
images_directory = '../Bilder/Fließtext'
output_directory = '../Modell_Output/Molmo/Fließtext'
durations_file_path = os.path.join(output_directory, "durations.txt")
os.makedirs(output_directory, exist_ok=True)

# Datei für Zeitmessungen vorbereiten
with open(durations_file_path, "w", encoding="utf-8") as durations_file:
    durations_file.write("Dateiname, Durchlaufzeit (Sekunden)\n")

# Alle Bilder im Verzeichnis durchlaufen
for image_filename in sorted(os.listdir(images_directory)):
    image_path = os.path.join(images_directory, image_filename)
    if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image = Image.open(image_path)

        start_time = time.time()  # Startzeit messen

        # Bildverarbeitung und Textgenerierung
        inputs = processor.process(images=[image], text="Please extract the text from the document provided. Please note that the text must be retained correctly and the formatting and layout must also be adopted.")
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=1024, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        #Please extract the text from the document provided. It contains a table, please adopt the layout and make sure that the value pairs are correct.

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Text speichern
        output_text_path = os.path.join(output_directory, os.path.splitext(image_filename)[0] + '.txt')
        with open(output_text_path, 'w', encoding='utf-8') as file:
            file.write(generated_text)

        duration = time.time() - start_time  # Durchlaufzeit berechnen
        with open(durations_file_path, "a", encoding="utf-8") as durations_file:
            durations_file.write(f"{image_filename}, {duration:.2f}\n")

        print(f"Text aus {image_filename} in {output_text_path} gespeichert. Durchlaufzeit: {duration:.2f} Sekunden")

        # Speicherbereinigung
        image.close()
        del inputs, output, generated_tokens, generated_text
