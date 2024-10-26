import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

image = Image.open('../testbilder/0.png').convert('RGB')

# 1. Lade das Modell und den Prozessor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 2. Definiere die Eingabedaten
# Stelle sicher, dass der Pfad zum Bild korrekt ist
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Extract text from whole png"},
        ],
    }
]

# 3. Bereite die Texteingaben vor
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Extrahiere die Bild- und Videoeingaben (falls vorhanden)
image_inputs, video_inputs = process_vision_info(messages)

# 5. Erstelle die Eingabetensoren für das Modell
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda" if torch.cuda.is_available() else "cpu")  # CUDA verwenden, falls verfügbar

# 6. Generiere die Ausgabe
#generated_ids = model.generate(**inputs, max_new_tokens=128)

# 6. Generiere die Ausgabe mit längerer Tokenlänge
generated_ids = model.generate(
    **inputs, 
    max_new_tokens=512,       # Maximale Tokenanzahl für längere Texte
           # Mindestlänge (optional)
    no_repeat_ngram_size=3,    # Verhindert Wiederholungen (optional)
    do_sample=False,
    num_beams=1,
    temperature=0.0
)

# 7. Bereinige die generierte Ausgabe
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]

# 8. Zeige das Ergebnis
print("Beschreibung des Bildes:", output_text)
