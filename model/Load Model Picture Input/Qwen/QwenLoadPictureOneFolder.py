import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

# Konfiguration
repo = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lade Modell und Prozessor
model = Qwen2VLForConditionalGeneration.from_pretrained(repo, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(repo)

# Verzeichnispfade
image_folder = "../Bilder/uneinheitliches Layout"  # Ordner mit Bildern
output_folder = "../Modell_Output/Qwen/uneinheitliches Layout"  # Ordner für die Ausgaben
os.makedirs(output_folder, exist_ok=True)

# Schleife über alle Bilder im Verzeichnis
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')  # Bild in RGB konvertieren

        # Definiere die Eingabedaten für jedes Bild neu
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Funktioniert oft: I cant read the text from the picture, please extract the complete text and the Characters. But please dont add any additional words"},
                ],
            }
        ]
        #Funktioniert oft: I cant read the text from the picture, please extract the complete text and the Characters. But please dont add any additional words
        #Please extract the text of eyery column and row of the provided table

        # Bereite die Texteingaben vor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Extrahiere die Bild- und Videoeingaben
        image_inputs, video_inputs = process_vision_info(messages)

        # Erstelle die Eingabetensoren für das Modell ohne `padding`-Argument
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(device)

        # Generiere die Ausgabe für das aktuelle Bild
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,       # Maximale Tokenanzahl für längere Texte
                min_length=100,            # Mindestlänge (optional)
                no_repeat_ngram_size=3,    # Verhindert Wiederholungen (optional)
                do_sample=False,
                temperature=0.0
            )

        # Bereinige die generierte Ausgabe
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Speichere die Modellausgabe sofort in einer Textdatei
        output_file = os.path.join(output_folder, f"{filename.split('.')[0]}_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        # Ausgabe bestätigen
        print(f"Text aus {filename} in {output_file} gespeichert.")

        # Speicherbereinigung: Bild schließen und Eingabedaten löschen
        image.close()
        del inputs, generated_ids, generated_ids_trimmed, output_text
