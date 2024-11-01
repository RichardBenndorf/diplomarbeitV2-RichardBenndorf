import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os

# Modell und Prozessor laden
processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto')
model = AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto')

# Bild aus lokalem Verzeichnis laden
image_path = os.path.join('../testbilder', '0.png')  # Ersetze 'dein_bild.jpg' mit dem Namen deines Bildes
image = Image.open(image_path)

# Bild verarbeiten
inputs = processor.process(images=[image], text="Only extract the text and do not add any additional words.")

# Modell für die Bildbeschreibung verwenden
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=1000, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# Ergebnis ausgeben
generated_tokens = output[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)
