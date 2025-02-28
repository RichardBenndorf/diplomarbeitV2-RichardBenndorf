import torch
import time
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import requests

# Konfiguration
model_id = "/mnt/extern/Moritz_LLM/LLAMA/model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lade Modell und Prozessor
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Verzeichnispfade
image_folder = "../Bilder/Tabellenformat"  # Ordner mit Bildern
output_folder = "../Modell_Output/Llama/Tabellenformat"  # Ordner für die Ausgaben
durations_path = os.path.join(output_folder, "durations.txt")  # Pfad für die Zeitmessungsdatei
os.makedirs(output_folder, exist_ok=True)

# Dummy-Bild laden, um das Modell zu initialisieren
dummy_image_path = "../DummyPicture/dummy.png"  # Stelle sicher, dass diese Bilddatei existiert
dummy_image = Image.open(dummy_image_path).convert('RGB')  # Öffne und konvertiere das Dummy-Bild
start_time = time.time()
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Please summarize the contents of the image."}
        ]
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    dummy_image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(device)
model.generate(
    **inputs, 
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
                    {"type": "image"},
                    {"type": "text", "text": "Please extract the text from the document provided. It contains a table, please adopt the layout and make sure that the value pairs are correct."},
                ],
            }
        ]
        #I have this old document. I need the text it contains for my work. Can you please extract the text for me, but please don't add any words. I need the text exactly as it is in the document. Pay attention to the formatting and the correct use of line breaks.
       #Please extract the text from the image. Please note that some table rows have a line break, please always display this correctly. Please display the specified format correctly. 
       # Please extract the text from the image. Make sure to maintain the table layout and maintain the correct text order. Please do not add any additional words.
       #Please extract the text from the image. Pay attention to the correct use of line breaks. Dont't add any additional characters, words or numbersand dont change the Layout.
        # Bereite die Texteingaben vor
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Erstelle die Eingabetensoren für das Modell
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        # Generiere die Ausgabe für das aktuelle Bild
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=2048
            )

        # Bereinige die generierte Ausgabe
        output_text = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

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
        del inputs, output, output_text

print(f"Modell Ladezeit: {model_load_duration:.2f} Sekunden")
