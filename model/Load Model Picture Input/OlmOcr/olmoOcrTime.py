import torch
import time
import os
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.prompts import build_finetuning_prompt

# Verzeichnispfade für Bilder und Ausgaben
image_folder = "../Bilder/Diagramme und infografische Elemente"  # Ordner mit Bildern
output_folder = "../Modell_Output/OlmOCR/Diagramme und infografische Elemente"  # Ordner für die Ausgaben
durations_path = os.path.join(output_folder, "durations.txt")  # Pfad für die Zeitmessungsdatei
os.makedirs(output_folder, exist_ok=True)

# Zeitmessung für Modell-Ladezeit
start_time = time.time()
model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("allenai/olmOCR-7B-0225-preview")  # OlmOCR Prozessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model_load_duration = time.time() - start_time  # Modellladezeit messen

# Speichere Modell-Ladezeit
with open(durations_path, "w", encoding="utf-8") as durations_file:
    durations_file.write(f"Modell Ladezeit: {model_load_duration:.2f} Sekunden\n")

# Schleife über alle Bilder im Verzeichnis
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert('RGB')  # Bild in RGB konvertieren

        start_time = time.time()  # Startzeit für die Bildverarbeitung messen

        # Optimierter Prompt für reine Textextraktion
        prompt = "Please extract the labels from the technical drawings. Please do not create a numbering if there is none in the image."
        #Extract only the raw text from the image. Ignore graphic illustrations, for me only the text is relevant KopfFußzeile
        #Extract only the raw text from the image. Preserve the original line breaks, spacing, and formatting exactly as in the document.
        # Eingabe für das Modell vorbereiten
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Text durch das Prozessor-Template laufen lassen
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for (key, value) in inputs.items()}

        # Generieren der Ausgabe
        output = model.generate(
            **inputs,
            temperature=0.0,  # Set to 0 to make the output deterministic
            max_new_tokens=2048,  # Allow more space for longer text
            num_return_sequences=1,
            do_sample=False,  # Disable randomness for precise extraction
        )

        # Ergebnis dekodieren und Zeilenumbrüche wiederherstellen
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        text_output = text_output.replace("\\n", "\n")  # Stellen sicher, dass Umbrüche richtig angezeigt werden

        # Zeitmessung abschließen
        processing_duration = time.time() - start_time

        # Speichern des Outputs
        image_output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_output.txt")
        with open(image_output_file, "w", encoding="utf-8") as f:
            f.write(text_output)

        # Speichern der Zeitmessung
        with open(durations_path, "a", encoding="utf-8") as durations_file:
            durations_file.write(f"{filename}, {processing_duration:.2f} Sekunden\n")

        print(f"Extrahierter Text aus {filename} gespeichert in: {image_output_file}")
        print(f"Verarbeitungszeit: {processing_duration:.2f} Sekunden")