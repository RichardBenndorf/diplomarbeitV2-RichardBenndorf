import difflib
from nltk.metrics import edit_distance
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
import os

def cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0]
    return cosine_sim

def evaluate_extraction(goldstandard_path, extracted_path):
    # Einlesen der Dateien
    with open(goldstandard_path, 'r', encoding='utf-8') as gs_file:
        goldstandard = gs_file.read().strip()

    with open(extracted_path, 'r', encoding='utf-8') as ex_file:
        extracted = ex_file.read().strip()

    goldstandard_words = goldstandard.split()
    extracted_words = extracted.split()

    # 1. Zeichenbasierte Metriken
    char_distance = edit_distance(goldstandard, extracted)
    char_error_rate = char_distance / len(goldstandard) if len(goldstandard) > 0 else 0
    relative_char_distance = char_distance / max(len(goldstandard), len(extracted)) if max(len(goldstandard), len(extracted)) > 0 else 0

    # 2. Wortbasierte Metriken
    word_distance = edit_distance(goldstandard_words, extracted_words)
    word_error_rate = word_distance / len(goldstandard_words) if len(goldstandard_words) > 0 else 0
    relative_word_distance = word_distance / max(len(goldstandard_words), len(extracted_words)) if max(len(goldstandard_words), len(extracted_words)) > 0 else 0

    # Anzahl korrekt erkannter Wörter (ohne Berücksichtigung der Reihenfolge)
    correct_words = sum(1 for word in extracted_words if word in goldstandard_words)
    total_words = len(goldstandard_words)
    correct_word_percentage = (correct_words / total_words) * 100 if total_words > 0 else 0

    # Fehlende und zusätzliche Wörter
    missing_words = [word for word in goldstandard_words if word not in extracted_words]
    extra_words = [word for word in extracted_words if word not in goldstandard_words]

    # Precision, Recall und F1-Score (auf Wortbasis)
    y_true = [1 if word in goldstandard_words else 0 for word in goldstandard_words]
    y_pred = [1 if word in goldstandard_words else 0 for word in extracted_words]

    # Angleichung der Längen von y_true und y_pred
    if len(y_true) < len(y_pred):
        y_true.extend([0] * (len(y_pred) - len(y_true)))
    elif len(y_pred) < len(y_true):
        y_pred.extend([0] * (len(y_true) - len(y_pred)))

    precision = precision_score(y_true, y_pred, zero_division=0, average='binary')
    recall = recall_score(y_true, y_pred, zero_division=0, average='binary')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='binary')

    # 3. Semantische Metriken
    # Cosine Similarity
    cosine_sim = cosine_similarity(goldstandard, extracted)

    # BLEU Score
    bleu_score = sentence_bleu([goldstandard_words], extracted_words)

    # 4. Ausgabe der Ergebnisse
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

Semantische Metriken:
- Cosine Similarity: {cosine_sim:.2%}
- BLEU Score: {bleu_score:.2%}

Weitere Metriken:
- Precision: {precision:.2%}
- Recall: {recall:.2%}
- F1-Score: {f1:.2%}
"""

    detailed_content = "Detaillierter Vergleich:\n-----------------------\n"
    diff = difflib.unified_diff(goldstandard_words, extracted_words, lineterm='')
    for line in diff:
        detailed_content += line + "\n"

    return summary_content, detailed_content

def process_all_files(goldstandard_directory="../../Load Model Picture Input/Goldstandard/Fließtext", extracted_directory="../../Load Model Picture Input/Modell_Output/Qwen/Fließtext", output_directory="Ergebnis_Qwen"):
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
