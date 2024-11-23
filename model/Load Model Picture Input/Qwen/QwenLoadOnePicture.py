import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import time

# Konfiguration
repo = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lade Modell und Prozessor
model = Qwen2VLForConditionalGeneration.from_pretrained(repo, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(repo)

# Verzeichnispfade
image_path = "../Bilder/uneinheitliches Layout/uneinheitliches_Layout_5.png"  # Pfad zum Bild
output_folder = "../Modell_Output/Qwen/uneinheitliches Layout"  # Ordner für die Ausgaben
dummy_image_path = "../DummyPicture/dummy.png"  # Pfad zur Dummy-Datei
os.makedirs(output_folder, exist_ok=True)

# Dummy-Bild laden, um das Modell zu initialisieren
dummy_image = Image.open(dummy_image_path).convert('RGB')
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
# Dummy-Durchlauf zur Modellinitialisierung
model.generate(**dummy_inputs, max_new_tokens=1)
model_load_duration = time.time() - start_time
dummy_image.close()

# Bild laden und in RGB konvertieren
image = Image.open(image_path).convert('RGB')

# Definiere die Eingabedaten für das Bild
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "I have received the following picture. I need all the words, numbers and characters it contains. It is important that the text is extracted 1:1. No characters may be added or omitted."},
        ],
    }
]#I have these very old text documents, which were written on a typewriter. I need the text they contain, but I can't read them. Can you please extract the text for me? Please make sure to include all paragraphs. 
# Bereite die Texteingaben vor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

# Erstelle die Eingabetensoren für das Modell
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt"
).to(device)

# Startzeit messen
start_time = time.time()

# Generiere die Ausgabe für das Bild
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1024,
        min_length=100,
        no_repeat_ngram_size=3,
        do_sample=False,
        temperature=0.0
    )

# Berechne die Durchlaufzeit
processing_time = time.time() - start_time

# Bereinige die generierte Ausgabe
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

# Speichere die Modellausgabe in einer Textdatei
output_file = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

# Ausgabe bestätigen
print(f"Text aus {image_path} in {output_file} gespeichert.")
print(f"Modell Ladezeit: {model_load_duration:.2f} Sekunden")
print(f"Durchlaufzeit: {processing_time:.2f} Sekunden")

# Speicherbereinigung: Bild schließen und Eingabedaten löschen
image.close()
del inputs, generated_ids, generated_ids_trimmed, output_text
