import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

# Konfiguration
repo = "Qwen/Qwen2-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lade Modell und Prozessor
model = Qwen2VLForConditionalGeneration.from_pretrained(repo, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(repo)

# Verzeichnispfade
image_folder = "../../Bilder/Kopf_Fußzeilen"  # Ordner mit Bildern
output_folder = "../../Modell_Output/Qwen7b/Kopf_Fußzeilen"  # Ordner für die Ausgaben
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
                    {"type": "text", "text": "I have this old document. I need the text it contains for my work. Can you please extract the text for me, but please don't add any words. I need the text exactly as it is in the document."},
                ],
            }
        ]
        #Funktioniert oft: I cant read the text from the picture, please extract the complete text and the Characters. But please dont add any additional words
        #Please extract the text of eyery column and row of the provided table
        #Schreibmaschinenschrift:
        #I have this old document. I need the text it contains for my work. Can you please extract the text for me, but please don't add any words. I need the text exactly as it is in the document. 
        #mehrspaltiges Layout
        #I have this old document. I need the text it contains for my work. Can you please give me the text contained in it in a tabular form. I need exactly the text that is in the picture. 
        #I have this old document. I need the text it contains for my work. Can you please give me the text contained in it in a tabular form. I need exactly the text that is in the picture. Please view the entire table.

        #Tabelle
        # Please extract the content of the provided table. für 4 und 5 gut
        #I have been given the task of copying the table from the old document. Can you please help me with this and take over the task. 5, 8 sehr gut

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
