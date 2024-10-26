# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/nougat-base")
model = AutoModel.from_pretrained("facebook/nougat-base")