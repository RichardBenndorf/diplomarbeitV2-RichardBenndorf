from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
image = Image.open('../testbilder/0.png').convert('RGB')

# Laden des vortrainierten Modells und Prozessors
model = Qwen2VLForConditionalGeneration.from_pretrained("BAAI/Aquila-VL-2B-llava-qwen", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("BAAI/Aquila-VL-2B-llava-qwen")

# Vorbereitung der Eingabe, inklusive einer URL zu einem Bild
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Vorbereitung für die Inferenz
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Verschieben der Daten auf eine GPU, falls verfügbar
if torch.cuda.is_available():
    inputs = inputs.to("cuda")

# Generierung der Ausgabe
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(output_text)
