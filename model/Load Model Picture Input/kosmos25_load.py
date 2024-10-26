import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Kosmos2_5ForConditionalGeneration, Kosmos2_5Processor

# Lade den Tokenizer
tokenizer = Kosmos2_5Processor.from_pretrained("microsoft/kosmos-2.5")

# Lade den heruntergeladenen Checkpoint (Modelldatei "ckpt.pt")
#checkpoint_path = "ckpt.pt"  # Stelle sicher, dass der Pfad korrekt ist

# Lade das Modell und verwende den heruntergeladenen Checkpoint
model = Kosmos2_5ForConditionalGeneration.from_pretrained("microsoft/kosmos-2.5")#, state_dict=torch.load(checkpoint_path))

print(model)





