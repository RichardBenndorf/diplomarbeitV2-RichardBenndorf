import os
import html
import re
from collections import Counter

def clean_html_entities(text):
    return html.unescape(text)

def clean_special_characters(text):
    # Entfernt alle Sonderzeichen außer einfachen Punkten und Kommata
    return re.sub(r'[^a-zA-Z0-9äöüÄÖÜß .,]', '', text)

def text_to_bow(text):
    # Verarbeitet den Text, um ein Bag-of-Words zu erstellen
    words = text.lower().split()
    return Counter(words)

def read_and_clean_text(filename):
    # Liest und bereinigt den Text von HTML-Entities und Sonderzeichen
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        text = clean_html_entities(text)
        return clean_special_characters(text)

def calculate_jaccard_index(bow1, bow2):
    # Berechnet den Jaccard-Index zwischen zwei Bag-of-Words
    common_words = bow1 & bow2
    total_words = bow1 | bow2
    return len(common_words) / len(total_words)

def calculate_differences(gold_bow, model_bow):
    # Berechnet die Unterschiede zwischen Goldstandard und Modellausgabe
    common_words = gold_bow & model_bow
    incorrect_words = model_bow - gold_bow
    missed_words = gold_bow - model_bow
    return common_words, incorrect_words, missed_words

def write_detailed_output(filename, jaccard_index, common_words, incorrect_words, missed_words):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        total_correct_words = sum(common_words.values())
        file.write(f"Zuordenbare Wörter: {total_correct_words}\n")
        sorted_common_words = sorted(common_words.items(), key=lambda item: item[1], reverse=True)
        for word, freq in sorted_common_words:
            file.write(f"{word}: {freq}\n")

        file.write("\nNicht zuordenbare Wörter:\n")
        for word, freq in incorrect_words.items():
            file.write(f"{word}: {freq}\n")

        file.write("\nFehlende Wörter:\n")
        for word, freq in missed_words.items():
            file.write(f"{word}: {freq}\n")

        file.write(f"\nJaccard-Index: {jaccard_index:.2%}\n")

# Pfad zu den Goldstandard- und Modell_Output-Ordnern
goldstandard_path = '../Load Model Picture Input/Goldstandard'
model_output_path = '../Load Model Picture Input/Modell_Output'
output_path = 'Evaluation of the Output/BoW_Detail'  # Ordner für die detaillierten Outputs'

model_files = [f for f in os.listdir(model_output_path) if f.endswith('.txt')]
gold_files = [f.replace('_output', '') for f in model_files]

results = {}
for gold_file, model_file in zip(gold_files, model_files):
    gold_text = read_and_clean_text(os.path.join(goldstandard_path, gold_file))
    model_text = read_and_clean_text(os.path.join(model_output_path, model_file))
    
    gold_bow = text_to_bow(gold_text)
    model_bow = text_to_bow(model_text)
    
    jaccard_index = calculate_jaccard_index(gold_bow, model_bow)
    common_words, incorrect_words, missed_words = calculate_differences(gold_bow, model_bow)
    
    detailed_filename = os.path.join(output_path, f"detail_{model_file[:-4]}.txt")
    write_detailed_output(detailed_filename, jaccard_index, common_words, incorrect_words, missed_words)
    
    results[model_file] = {'jaccard_index': jaccard_index, 'detail_file': detailed_filename}

summary_filename = os.path.join(output_path, 'evaluation_summary.txt')
with open(summary_filename, 'w', encoding='utf-8') as file:
    for filename, data in results.items():
        file.write(f"{filename}: Jaccard-Index = {data['jaccard_index']:.2%}\n")

for filename, data in results.items():
    print(f"{filename}: Detailed output written to {data['detail_file']}")
print(f"Summary of results written to {summary_filename}")
