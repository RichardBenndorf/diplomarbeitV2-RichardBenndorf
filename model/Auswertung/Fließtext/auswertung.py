import difflib
from nltk.metrics import edit_distance
from sklearn.metrics import precision_score, recall_score, f1_score
import os

def evaluate_extraction(goldstandard_path, extracted_path):
    # Einlesen der Dateien
    with open(goldstandard_path, 'r', encoding='utf-8') as gs_file:
        goldstandard = gs_file.read().strip().split()

    with open(extracted_path, 'r', encoding='utf-8') as ex_file:
        extracted = ex_file.read().strip().split()

    # 1. Zeichenbasierte Metriken
    gs_text = ' '.join(goldstandard)
    ex_text = ' '.join(extracted)
    char_distance = edit_distance(gs_text, ex_text)
    char_error_rate = char_distance / len(gs_text) if len(gs_text) > 0 else 0
    relative_char_distance = char_distance / max(len(gs_text), len(ex_text)) if max(len(gs_text), len(ex_text)) > 0 else 0

    # 2. Wortbasierte Metriken
    word_distance = edit_distance(goldstandard, extracted)
    word_error_rate = word_distance / len(goldstandard) if len(goldstandard) > 0 else 0
    relative_word_distance = word_distance / max(len(goldstandard), len(extracted)) if max(len(goldstandard), len(extracted)) > 0 else 0

    # Anzahl korrekt erkannter Wörter (ohne Berücksichtigung der Reihenfolge)
    correct_words = sum(1 for word in extracted if word in goldstandard)
    total_words = len(goldstandard)
    correct_word_percentage = (correct_words / total_words) * 100 if total_words > 0 else 0

    # Fehlende und zusätzliche Wörter
    missing_words = [word for word in goldstandard if word not in extracted]
    extra_words = [word for word in extracted if word not in goldstandard]

    # Precision, Recall und F1-Score (auf Wortbasis)
    y_true = [1 if word in goldstandard else 0 for word in goldstandard]
    y_pred = [1 if word in goldstandard else 0 for word in extracted]

    # Angleichung der Längen von y_true und y_pred
    if len(y_true) < len(y_pred):
        y_true.extend([0] * (len(y_pred) - len(y_true)))
    elif len(y_pred) < len(y_true):
        y_pred.extend([0] * (len(y_true) - len(y_pred)))

    precision = precision_score(y_true, y_pred, zero_division=0, average='binary')
    recall = recall_score(y_true, y_pred, zero_division=0, average='binary')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='binary')

    # 3. Ausgabe der Ergebnisse
    summary_content = f"""
Zusammenfassung der Metriken:
--------------------------------
Zeichenbasierte Metriken:
- Levenshtein-Distanz (Zeichen): {char_distance}
- Relative Levenshtein-Distanz (Zeichen): {relative_char_distance:.2%}
- Character Error Rate (CER): {char_error_rate:.2%}

Wortbasierte Metriken:
- Levenshtein-Distanz (Wörter): {word_distance}
- Relative Levenshtein-Distanz (Wörter): {relative_word_distance:.2%}
- Word Error Rate (WER): {word_error_rate:.2%}
- Anzahl korrekt erkannter Wörter: {correct_words} / {total_words} ({correct_word_percentage:.2f}%)
- Fehlende Wörter: {len(missing_words)}
- Zusätzliche Wörter: {len(extra_words)}

Weitere Metriken:
- Precision: {precision:.2%}
- Recall: {recall:.2%}
- F1-Score: {f1:.2%}
"""

    detailed_content = "Detaillierter Vergleich:\n-----------------------\n"
    diff = difflib.unified_diff(goldstandard, extracted, lineterm='')
    for line in diff:
        detailed_content += line + "\n"

    return summary_content, detailed_content

def process_all_files(goldstandard_directory="../../Load Model Picture Input/Goldstandard/Fließtext", extracted_directory="../../Load Model Picture Input/Modell_Output/Qwen7b/Fließtext", output_directory="../../Ergebnis"):
    # Sicherstellen, dass das Ausgabe-Verzeichnis existiert
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(goldstandard_directory):
        if file_name.startswith("Goldstandard_") and file_name.endswith(".txt"):
            index = file_name.split("_")[1].split(".")[0]
            goldstandard_path = os.path.join(goldstandard_directory, file_name)
            extracted_path = os.path.join(extracted_directory, f"Fließtext_{index}_output.txt")

            if os.path.exists(extracted_path):
                summary, details = evaluate_extraction(goldstandard_path, extracted_path)

                # Ergebnisse speichern
                summary_path = os.path.join(output_directory, f"Summary_{index}.txt")
                details_path = os.path.join(output_directory, f"Details_{index}.txt")

                # Überschreiben vorhandener Dateien
                with open(summary_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(summary)

                with open(details_path, 'w', encoding='utf-8') as details_file:
                    details_file.write(details)

                print(f"Auswertung für Datei {index} abgeschlossen. Ergebnisse gespeichert.")
            else:
                print(f"Passende Datei für {file_name} nicht gefunden.")

# Beispielaufruf
process_all_files()
