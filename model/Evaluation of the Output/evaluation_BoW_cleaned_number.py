import os
import html
import re
from collections import Counter

def clean_html_entities(text):
    return html.unescape(text)

def add_space_around_numbers_and_special_chars(text):
    # Fügt Leerzeichen zwischen Zahlen/Sonderzeichen und Wörtern hinzu
    text = re.sub(r'(\d+)([^\d\s])', r'\1 \2', text)  # Zahl gefolgt von einem Buchstaben/Sonderzeichen
    text = re.sub(r'([^\d\s])(\d+)', r'\1 \2', text)  # Buchstabe/Sonderzeichen gefolgt von einer Zahl
    return text

def remove_numbers(text):
    # Entfernt alle Zahlen aus dem Text
    return re.sub(r'\d+', '', text)

def clean_special_characters(text):
    # Entfernt unerwünschte Sonderzeichen, außer Punkte und Kommas
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß .,]', '', text)
    return text

def text_to_bow(text):
    # Verarbeitet den Text, um ein Bag-of-Words zu erstellen
    words = text.lower().split()
    return Counter(words)

def read_and_clean_text(filename):
    # Liest und bereinigt den Text von HTML-Entities, Zahlen und Sonderzeichen
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        text = clean_html_entities(text)
        text = add_space_around_numbers_and_special_chars(text)
        text = remove_numbers(text)  # Zahlen entfernen
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
goldstandard_path = '../Load Model Picture Input/Goldstandard/Fließtext'
model_output_path = '../Load Model Picture Input/Modell_Output/Qwen7b/Fließtext'
output_path = 'Evaluation of the Output/BoW_Detail/Fließtext/Cleaned_Number'  # Ordner für die detaillierten Outputs

# Dateien filtern und Mappen erstellen
model_files = [f for f in os.listdir(model_output_path) if f.startswith('Fließtext_') and f.endswith('_output.txt')]
gold_files = [f for f in os.listdir(goldstandard_path) if f.startswith('Goldstandard_') and f.endswith('.txt')]

# Mapping anhand der Nummern aus den Dateinamen erstellen
gold_to_model_mapping = {}
for model_file in model_files:
    model_number = model_file.split('_')[1]  # Extrahiert die Nummer aus 'Fließtext_1_output.txt'
    gold_file = f"Goldstandard_{model_number}.txt"
    if gold_file in gold_files:
        gold_to_model_mapping[gold_file] = model_file

# Ergebnisse berechnen
results = {}
for gold_file, model_file in gold_to_model_mapping.items():
    gold_text = read_and_clean_text(os.path.join(goldstandard_path, gold_file))
    model_text = read_and_clean_text(os.path.join(model_output_path, model_file))
    
    gold_bow = text_to_bow(gold_text)
    model_bow = text_to_bow(model_text)
    
    jaccard_index = calculate_jaccard_index(gold_bow, model_bow)
    common_words, incorrect_words, missed_words = calculate_differences(gold_bow, model_bow)
    
    detailed_filename = os.path.join(output_path, f"detail_{model_file[:-4]}.txt")
    write_detailed_output(detailed_filename, jaccard_index, common_words, incorrect_words, missed_words)
    
    results[model_file] = {'jaccard_index': jaccard_index, 'detail_file': detailed_filename}

# Zusammenfassung schreiben
summary_filename = os.path.join(output_path, 'evaluation_summary.txt')
with open(summary_filename, 'w', encoding='utf-8') as file:
    for filename, data in results.items():
        file.write(f"{filename}: Jaccard-Index = {data['jaccard_index']:.2%}\n")

# Ausgabe auf der Konsole
for filename, data in results.items():
    print(f"{filename}: Detailed output written to {data['detail_file']}")
print(f"Summary of results written to {summary_filename}")
