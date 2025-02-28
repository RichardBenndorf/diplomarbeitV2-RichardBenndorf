import torch
import time
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
image_folder = "../../Bilder/Tabellenformat"  # Ordner mit Bildern
output_folder = "../../Modell_Output/Qwen7b/Tabellenformat"  # Ordner für die Ausgaben
durations_path = os.path.join(output_folder, "durations.txt")  # Pfad für die Zeitmessungsdatei
os.makedirs(output_folder, exist_ok=True)

# Dummy-Bild laden, um das Modell zu initialisieren
dummy_image_path = "../../DummyPicture/dummy.png"  # Stelle sicher, dass diese Bilddatei existiert
dummy_image = Image.open(dummy_image_path).convert('RGB')  # Öffne und konvertiere das Dummy-Bild
start_time = time.time()
dummy_messages = [{"role": "user", "content": [{"type": "image", "image": dummy_image}]}]
dummy_text = processor.apply_chat_template(dummy_messages, tokenize=False, add_generation_prompt=True)
dummy_image_inputs, dummy_video_inputs = process_vision_info(dummy_messages)
dummy_inputs = processor(
    text=[dummy_text],
    images=dummy_image_inputs,
    videos=dummy_video_inputs,
    return_tensors="pt"
).to(device)
model.generate(
    **dummy_inputs, 
    max_new_tokens=1
)
model_load_duration = time.time() - start_time  # Modellladezeit messen
dummy_image.close()  # Dummy-Bild schließen

# Datei für Zeitmessungen vorbereiten
with open(durations_path, "w", encoding="utf-8") as durations_file:
    durations_file.write(f"Modell Ladezeit, {model_load_duration:.2f} Sekunden\n")
    durations_file.write("Dateiname, Durchlaufzeit (Sekunden)\n")


# Schleife über alle Bilder im Verzeichnis
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')  # Bild in RGB konvertieren

        start_time = time.time()  # Startzeit für die Bildverarbeitung messen

        # Definiere die Eingabedaten für jedes Bild neu
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please extract the text from the document provided. It contains a table, please adopt the layout and make sure that the value pairs are correct."},
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
        
        #uneinheitliches Layout
        #I have this old document. I need the text it contains for my work. Can you please extract the text for me, but please don't add any words. I need the text exactly as it is in the document. Pay attention to the formatting and the correct use of line breaks.

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

        duration = time.time() - start_time  # Durchlaufzeit berechnen
        with open(durations_path, "a", encoding="utf-8") as durations_file:
            durations_file.write(f"{filename}, {duration:.2f}\n")

        # Ausgabe bestätigen
        print(f"Text aus {filename} in {output_file} gespeichert. Durchlaufzeit: {duration:.2f} Sekunden")

        # Speicherbereinigung: Bild schließen und Eingabedaten löschen
        image.close()
        del inputs, generated_ids, generated_ids_trimmed, output_text
print(f"Modell Ladezeit: {model_load_duration:.2f} Sekunden")