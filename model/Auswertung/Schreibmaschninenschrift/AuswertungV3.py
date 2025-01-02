import difflib
from nltk.metrics import edit_distance
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu
import os
from collections import Counter
import pandas as pd
import re

def clean_text(text):
    # Entfernt alle nicht-alphanumerischen Zeichen (außer Leerzeichen)
    return re.sub(r'[^\w\s]', '', text)

def cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0]
    return cosine_sim

def find_near_matches(gold_words, extracted_words, threshold=2):
    near_matches = []
    for gold_word in gold_words:
        for extracted_word in extracted_words:
            if edit_distance(gold_word, extracted_word) <= threshold:
                near_matches.append((gold_word, extracted_word))
                extracted_words.remove(extracted_word)  # Verhindert Doppelzählung
                break
    return near_matches

def evaluate_extraction(goldstandard_path, extracted_path):
    # Einlesen der Dateien
    with open(goldstandard_path, 'r', encoding='utf-8') as gs_file:
        goldstandard = gs_file.read().strip()

    with open(extracted_path, 'r', encoding='utf-8') as ex_file:
        extracted = ex_file.read().strip()

    # Bereinigung der Texte
    goldstandard = clean_text(goldstandard)
    extracted = clean_text(extracted)

    goldstandard_words = goldstandard.split()
    extracted_words = extracted.split()

    # 1. Zeichenbasierte Metriken
    char_distance = edit_distance(goldstandard, extracted)
    char_error_rate = char_distance / len(goldstandard) if len(goldstandard) > 0 else 0

    # 2. Wortbasierte Metriken
    word_distance = edit_distance(goldstandard_words, extracted_words)
    word_error_rate = word_distance / len(goldstandard_words) if len(goldstandard_words) > 0 else 0

    # Häufigkeiten der Wörter berechnen
    goldstandard_word_count = Counter(goldstandard_words)
    extracted_word_count = Counter(extracted_words)

    # Anzahl korrekt erkannter Wörter berechnen (unter Berücksichtigung der Häufigkeiten)
    correct_words = sum(min(goldstandard_word_count[word], extracted_word_count[word]) for word in extracted_word_count)
    total_words = len(goldstandard_words)

    # Nahezu korrekt erkannte Wörter (Schreibfehler)
    near_matches = find_near_matches(goldstandard_words, extracted_words[:], threshold=2)
    fast_correct_words = len(near_matches)

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

    summary_content = f"""
Zusammenfassung der Metriken:
--------------------------------
Zeichenbasierte Metriken:
- Gesamtzeichen: {len(goldstandard)}
- Character Error Rate (CER): {char_error_rate * 100:.2f}%
- Falsche Zeichen (absolut): {char_distance}

Wortbasierte Metriken:
- Gesamtwörter: {total_words}
- Word Error Rate (WER): {word_error_rate * 100:.2f}%
- Korrekt erkannte Wörter: {correct_words}
- Fast korrekt erkannte Wörter: {fast_correct_words}
- Fehlende Wörter: {len(missing_words)}
- Zusätzliche Wörter: {len(extra_words)}

Semantische Metriken:
- Cosine Similarity: {cosine_sim * 100:.2f}%
- BLEU Score: {bleu_score * 100:.2f}%

Weitere Metriken:
- Precision: {precision:.2f}
- Recall: {recall:.2f}
- F1-Score: {f1:.2f}
"""

    detailed_comparison = []
    for word in goldstandard_words:
        if word in extracted_words:
            detailed_comparison.append(f"KORREKT: {word}")
        elif any(edit_distance(word, ew) <= 2 for ew in extracted_words):
            detailed_comparison.append(f"FAST KORREKT: {word}")
        else:
            detailed_comparison.append(f"FEHLT: {word}")

    for word in extracted_words:
        if word not in goldstandard_words:
            detailed_comparison.append(f"ZUSÄTZLICH: {word}")

    detailed_content = "\n".join(detailed_comparison)

    return summary_content, detailed_content, {
        "Gesamtzeichen": len(goldstandard),
        "Gesamtwörter": total_words,
        "CER": char_error_rate * 100,
        "Falsche Zeichen (absolut)": char_distance,
        "WER": word_error_rate * 100,
        "Korrekt erkannte Wörter": correct_words,
        "Fast korrekt erkannte Wörter": fast_correct_words,
        "Fehlende Wörter": len(missing_words),
        "Zusätzliche Wörter": len(extra_words),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Cosine Similarity": cosine_sim * 100,
        "BLEU Score": bleu_score * 100
    }

def process_all_files_to_excel(goldstandard_directory, extracted_directory, output_directory, excel_path):
    # Sicherstellen, dass das Ausgabe-Verzeichnis existiert
    os.makedirs(output_directory, exist_ok=True)

    # Ergebnisse sammeln
    results = []
    summary_contents = []
    detailed_contents = []

    for file_name in os.listdir(goldstandard_directory):
        if file_name.startswith("Goldstandard_") and file_name.endswith(".txt"):
            index = file_name.split("_")[1].split(".")[0]
            goldstandard_path = os.path.join(goldstandard_directory, file_name)
            extracted_path = os.path.join(extracted_directory, f"schreibmaschinenschrift_{index}_output.txt")

            if os.path.exists(extracted_path):
                summary, details, metrics = evaluate_extraction(goldstandard_path, extracted_path)

                # Ergebnisse sammeln
                results.append({
                    "Durchlauf": index,
                    "Zeichenbasiert::Gesamtzeichen": metrics["Gesamtzeichen"],
                    "Zeichenbasiert::CER": metrics["CER"],
                    "Zeichenbasiert::Falsche Zeichen (absolut)": metrics["Falsche Zeichen (absolut)"],
                    "Wortbasiert::Gesamtwörter": metrics["Gesamtwörter"],
                    "Wortbasiert::WER": metrics["WER"],
                    "Wortbasiert::Korrekt erkannte Wörter": metrics["Korrekt erkannte Wörter"],
                    "Wortbasiert::Fast korrekt erkannte Wörter": metrics["Fast korrekt erkannte Wörter"],
                    "Wortbasiert::Fehlende Wörter": metrics["Fehlende Wörter"],
                    "Wortbasiert::Zusätzliche Wörter": metrics["Zusätzliche Wörter"],
                    "Weitere Metriken::Precision": metrics["Precision"],
                    "Weitere Metriken::Recall": metrics["Recall"],
                    "Weitere Metriken::F1 Score": metrics["F1 Score"],
                    "Semantisch::Cosine Similarity": metrics["Cosine Similarity"],
                    "Semantisch::BLEU Score": metrics["BLEU Score"]
                })

                summary_contents.append(f"Durchlauf {index}:\n{summary}\n")
                detailed_contents.append(f"Durchlauf {index}:\n{details}\n")

                print(f"Auswertung für Datei {index} abgeschlossen.")
            else:
                print(f"Passende Datei für {file_name} nicht gefunden. Erzeugter Pfad: {extracted_path}")

    # Ergebnisse in eine Excel-Datei schreiben
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Ergebnisse wurden in {excel_path} gespeichert.")

    # Zusammenfassung und Details schreiben
    summary_path = os.path.join(output_directory, "Summary_All.txt")
    details_path = os.path.join(output_directory, "Details_All.txt")

    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.writelines(summary_contents)

    with open(details_path, 'w', encoding='utf-8') as details_file:
        details_file.writelines(detailed_contents)

    print(f"Zusammenfassung und Details wurden gespeichert.")

# Beispielaufruf
process_all_files_to_excel(
    goldstandard_directory="../../Load Model Picture Input/Goldstandard/Schreibmaschinenschrift",
    extracted_directory="../../Load Model Picture Input/Modell_Output/Qwen7b/Schreibmaschinenschrift",
    output_directory="Ergebnis_Qwen7b",
    excel_path="Ergebnis_Qwen7b/results.xlsx"
)
