import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

# Configuration
repo = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(repo, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(repo)

# Define directories
image_folder = "../Bilder"  # Root folder with subdirectories containing images
output_folder = "../Modell_Output/Qwen"  # Output folder for text files
os.makedirs(output_folder, exist_ok=True)

# Traverse through all subdirectories in image_folder
for root, dirs, files in os.walk(image_folder):
    for filename in files:
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # Get full path of the image
            image_path = os.path.join(root, filename)
            image = Image.open(image_path).convert('RGB')  # Convert image to RGB

            # Prepare input data for each image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Extract text"},
                    ],
                }
            ]

            # Prepare text inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Process image and video inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Create input tensors for the model without `padding` argument
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            ).to(device)

            # Generate output for the current image
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=512,       # Maximum token count for longer text
                    min_length=100,           # Minimum length (optional)
                    no_repeat_ngram_size=3,   # Prevent repetition (optional)
                    do_sample=False
                )

            # Clean up generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Determine relative path for output file
            relative_path = os.path.relpath(root, image_folder)
            output_dir = os.path.join(output_folder, relative_path)
            os.makedirs(output_dir, exist_ok=True)

            # Save output text to file with the same name as the image
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)

            # Confirm output
            print(f"Text from {filename} saved to {output_file}.")

            # Clean up: Close image and delete input data
            image.close()
            del inputs, generated_ids, generated_ids_trimmed, output_text
