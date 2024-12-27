import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import time

# Konfiguration
repo = "Qwen/Qwen2-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lade Modell und Prozessor
model = Qwen2VLForConditionalGeneration.from_pretrained(repo, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(repo)

# Verzeichnispfade
image_path = "../../Bilder/Fließtext/Fließtext_2.png"  # Pfad zum Bild
output_folder = "../Modell_Output/Qwen7b/Fließtext"  # Ordner für die Ausgaben
os.makedirs(output_folder, exist_ok=True)

# Bild laden und in RGB konvertieren
image = Image.open(image_path).convert('RGB')

# Definiere die Eingabedaten für das Bild
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Please extract the text from the provided document, image document."},
        ],
    }
]

# Bereite die Texteingaben vor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

# Erstelle die Eingabetensoren für das Modell
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(device)

# Startzeit messen
start_time = time.time()

# Generiere die Ausgabe für das Bild
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        no_repeat_ngram_size=3,
        do_sample=False,
        num_beams=1,
        temperature=0.0
    )

# Berechne die Durchlaufzeit
processing_time = time.time() - start_time

# Bereinige die generierte Ausgabe
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]

# Speichere die Modellausgabe in einer Textdatei
output_file = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

# Ausgabe bestätigen
print(f"Text aus {image_path} in {output_file} gespeichert.")
print(f"Durchlaufzeit: {processing_time:.2f} Sekunden")

# Speicherbereinigung: Bild schließen
image.close()
