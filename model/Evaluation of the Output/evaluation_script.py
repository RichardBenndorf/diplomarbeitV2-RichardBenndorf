import os
from difflib import SequenceMatcher

# Verzeichnispfade
output_folder = "../Load Model Picture Input/Modell_Output"
goldstandard_folder = "../Load Model Picture Input/Goldstandard"
evaluation_file = "evaluation_summary.txt"

# Funktion zur Berechnung der Satzähnlichkeit mit SequenceMatcher
def sentence_similarity(sentence1, sentence2):
    return SequenceMatcher(None, sentence1, sentence2).ratio()

# Datei öffnen, um die Evaluierungsergebnisse zu speichern
with open(evaluation_file, "w", encoding="utf-8") as eval_file:
    # Schleife über alle Modellausgabe-Dateien
    for filename in sorted(os.listdir(output_folder)):
        if filename.endswith("_output.txt"):
            # Lade die Modellausgabe
            output_path = os.path.join(output_folder, filename)
            with open(output_path, "r", encoding="utf-8") as f:
                model_output = f.read().strip()

            # Lade den Goldstandard
            goldstandard_file = filename.replace("_output.txt", ".txt")
            goldstandard_path = os.path.join(goldstandard_folder, goldstandard_file)
            if os.path.exists(goldstandard_path):
                with open(goldstandard_path, "r", encoding="utf-8") as f:
                    goldstandard = f.read().strip()
            else:
                print(f"Goldstandard für {filename} nicht gefunden.")
                continue

            # Textzeilen aufteilen
            model_lines = model_output.split('\n')
            goldstandard_lines = goldstandard.split('\n')

            # Berechne die Satzähnlichkeit für jede Zeile
            total_similarity = 0
            num_lines = min(len(model_lines), len(goldstandard_lines))

            for model_line, gold_line in zip(model_lines, goldstandard_lines):
                similarity = sentence_similarity(model_line, gold_line)
                total_similarity += similarity

            # Berechne die Durchschnittsgenauigkeit für dieses Bild
            avg_similarity = total_similarity / num_lines if num_lines > 0 else 0

            # Speichere die Bewertung in der Evaluationsdatei
            eval_file.write(f"{filename.replace('_output.txt', '')}: Genauigkeit = {avg_similarity:.2f}\n")
            print(f"{filename.replace('_output.txt', '')}: Genauigkeit = {avg_similarity:.2f}")
