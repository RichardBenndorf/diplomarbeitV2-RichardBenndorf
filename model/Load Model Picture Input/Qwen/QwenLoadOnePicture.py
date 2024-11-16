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

# Bildpfad und Ausgabeordner
image_path = "../Bilder/Fließtext/Fließtext_5.png"  # Pfad zum Bild
output_folder = "../Modell_Output/Qwen/Fließtext"  # Ordner für die Ausgaben
os.makedirs(output_folder, exist_ok=True)

# Bild laden und in RGB konvertieren
image = Image.open(image_path).convert('RGB')

# Definiere die Eingabedaten für das Bild
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Can you please copy the text section written in the picture for me. I need it for my work. "},
        ],
    }
]
#I have this old document. Can you please copy it from the document. Please make sure that the multi-column layout is displayed correctly.
#I have this old document. Can you please copy it from the document. Please make sure that the multi-column layout is displayed correctly and that all entries in the columns are included.
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

# Generiere die Ausgabe für das Bild
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

# Speichere die Modellausgabe in einer Textdatei
output_file = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

# Ausgabe bestätigen
print(f"Text aus {image_path} in {output_file} gespeichert.")

# Speicherbereinigung: Bild schließen und Eingabedaten löschen
image.close()
del inputs, generated_ids, generated_ids_trimmed, output_text
