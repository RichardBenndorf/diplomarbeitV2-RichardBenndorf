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
                extracted_words.remove(extracted_word)
                break
    return near_matches

def evaluate_extraction(goldstandard_path, extracted_path):
    with open(goldstandard_path, 'r', encoding='utf-8') as gs_file:
        goldstandard = gs_file.read().strip()

    with open(extracted_path, 'r', encoding='utf-8') as ex_file:
        extracted = ex_file.read().strip()

    goldstandard = clean_text(goldstandard)
    extracted = clean_text(extracted)

    goldstandard_words = goldstandard.split()
    extracted_words = extracted.split()

    # Zeichenbasierte Metriken basierend auf Häufigkeiten
    goldstandard_char_count = Counter(goldstandard)
    extracted_char_count = Counter(extracted)

    char_correct = sum(min(goldstandard_char_count[char], extracted_char_count[char]) for char in goldstandard_char_count)
    char_missing = sum(max(goldstandard_char_count[char] - extracted_char_count.get(char, 0), 0) for char in goldstandard_char_count)
    char_extra = sum(max(extracted_char_count[char] - goldstandard_char_count.get(char, 0), 0) for char in extracted_char_count)
    cer = (char_missing + char_extra) / len(goldstandard) if len(goldstandard) > 0 else 0

    # Wortbasierte Metriken basierend auf Häufigkeiten
    goldstandard_word_count = Counter(goldstandard_words)
    extracted_word_count = Counter(extracted_words)

    word_correct = sum(min(goldstandard_word_count[word], extracted_word_count[word]) for word in goldstandard_word_count)
    word_missing = sum(max(goldstandard_word_count[word] - extracted_word_count.get(word, 0), 0) for word in goldstandard_word_count)
    word_extra = sum(max(extracted_word_count[word] - goldstandard_word_count.get(word, 0), 0) for word in extracted_word_count)

    # Word Error Rate (WER)
    wer = (word_missing + word_extra) / len(goldstandard_words) if len(goldstandard_words) > 0 else 0

    # Korrekt erkannte Quote
    correct_word_ratio = word_correct / len(goldstandard_words) if len(goldstandard_words) > 0 else 0

    # Fast korrekt erkannte Wörter
    near_matches = find_near_matches(
        [word for word in goldstandard_words if goldstandard_word_count[word] > extracted_word_count.get(word, 0)],
        [word for word in extracted_words if extracted_word_count[word] > goldstandard_word_count.get(word, 0)],
        threshold=2
    )
    fast_correct_words = len(near_matches)

    # Wörter, die bereits als fast korrekt erkannt wurden, entfernen
    near_match_words = [match[0] for match in near_matches]

    missing_words = [word for word in goldstandard_words if word not in extracted_words and word not in near_match_words]
    extra_words = [word for word in extracted_words if word not in goldstandard_words and word not in [match[1] for match in near_matches]]

    # Korrekte Wörter inkl. fast korrekter Wörter
    word_correct_with_near = word_correct + fast_correct_words

    # WER inkl. fast korrekter Wörter
    wer_with_near = (len(goldstandard_words) - word_correct_with_near) / len(goldstandard_words) if len(goldstandard_words) > 0 else 0

    # Korrekt erkannte Quote inkl. fast korrekter Wörter
    correct_word_ratio_with_near = word_correct_with_near / len(goldstandard_words) if len(goldstandard_words) > 0 else 0

    y_true = [1 if word in goldstandard_words else 0 for word in goldstandard_words]
    y_pred = [1 if word in goldstandard_words else 0 for word in extracted_words]

    if len(y_true) < len(y_pred):
        y_true.extend([0] * (len(y_pred) - len(y_true)))
    elif len(y_pred) < len(y_true):
        y_pred.extend([0] * (len(y_true) - len(y_pred)))

    precision = precision_score(y_true, y_pred, zero_division=0, average='binary')
    recall = recall_score(y_true, y_pred, zero_division=0, average='binary')
    f1 = f1_score(y_true, y_pred, zero_division=0, average='binary')

    cosine_sim = cosine_similarity(goldstandard, extracted)
    bleu_score = sentence_bleu([goldstandard_words], extracted_words)

    summary_content = f"""
Zusammenfassung der Metriken:
--------------------------------
Zeichenbasierte Metriken:
- Gesamtzeichen: {len(goldstandard)}
- Korrekte Zeichen: {char_correct}
- Fehlende Zeichen: {char_missing}
- Zusätzliche Zeichen: {char_extra}
- Character Error Rate (CER): {cer * 100:.2f}%

Wortbasierte Metriken:
- Gesamtwörter: {len(goldstandard_words)}
- Korrekt erkannte Wörter: {word_correct}
- Korrekt erkannte Wörter inkl. fast korrekter Wörter: {word_correct_with_near}
- Fehlende Wörter: {len(missing_words)}
- Zusätzliche Wörter: {len(extra_words)}
- Word Error Rate (WER): {wer * 100:.2f}%
- Word Error Rate inkl. fast korrekter Wörter: {wer_with_near * 100:.2f}%
- Fast korrekt erkannte Wörter: {fast_correct_words}
- Korrekt erkannte Quote: {correct_word_ratio * 100:.2f}%
- Korrekt erkannte Quote inkl. fast korrekter Wörter: {correct_word_ratio_with_near * 100:.2f}%

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
        elif word in near_match_words:
            detailed_comparison.append(f"FAST KORREKT: {word}")
        else:
            detailed_comparison.append(f"FEHLT: {word}")

    for word in extracted_words:
        if word not in goldstandard_words and word not in [match[1] for match in near_matches]:
            detailed_comparison.append(f"ZUSÄTZLICH: {word}")

    detailed_content = "\n".join(detailed_comparison)

    return summary_content, detailed_content, {
        "Gesamtzeichen": len(goldstandard),
        "Korrekte Zeichen": char_correct,
        "Fehlende Zeichen": char_missing,
        "Zusätzliche Zeichen": char_extra,
        "CER": cer * 100,
        "Gesamtwörter": len(goldstandard_words),
        "Korrekt erkannte Wörter": word_correct,
        "Korrekt erkannte Wörter inkl. fast korrekter Wörter": word_correct_with_near,
        "Fehlende Wörter": len(missing_words),
        "Zusätzliche Wörter": len(extra_words),
        "WER": wer * 100,
        "Word Error Rate inkl. fast korrekter Wörter": wer_with_near * 100,
        "Fast korrekt erkannte Wörter": fast_correct_words,
        "Korrekt erkannte Quote": correct_word_ratio * 100,
        "Korrekt erkannte Quote inkl. fast korrekter Wörter": correct_word_ratio_with_near * 100,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Cosine Similarity": cosine_sim * 100,
        "BLEU Score": bleu_score * 100
    }

def process_all_files_to_excel(goldstandard_directory, extracted_directory, output_directory, excel_path):
    os.makedirs(output_directory, exist_ok=True)

    results = []
    summary_contents = []
    detailed_contents = []

    for file_name in os.listdir(goldstandard_directory):
        if file_name.startswith("Goldstandard_") and file_name.endswith(".txt"):
            index = file_name.split("_")[1].split(".")[0]
            goldstandard_path = os.path.join(goldstandard_directory, file_name)
            extracted_path = os.path.join(extracted_directory, f"Diagramme_und_infografische_Elemente_{index}_output.txt")

            if os.path.exists(extracted_path):
                summary, details, metrics = evaluate_extraction(goldstandard_path, extracted_path)

                results.append({
                    "Durchlauf": index,
                    "Zeichenbasiert::Gesamtzeichen": metrics["Gesamtzeichen"],
                    "Zeichenbasiert::Korrekte Zeichen": metrics["Korrekte Zeichen"],
                    "Zeichenbasiert::Fehlende Zeichen": metrics["Fehlende Zeichen"],
                    "Zeichenbasiert::Zusätzliche Zeichen": metrics["Zusätzliche Zeichen"],
                    "Zeichenbasiert::CER": metrics["CER"],
                    "Wortbasiert::Gesamtwörter": metrics["Gesamtwörter"],
                    "Wortbasiert::Korrekt erkannte Wörter": metrics["Korrekt erkannte Wörter"],
                    "Wortbasiert::Korrekt erkannte Wörter inkl. fast korrekter Wörter": metrics["Korrekt erkannte Wörter inkl. fast korrekter Wörter"],
                    "Wortbasiert::Fehlende Wörter": metrics["Fehlende Wörter"],
                    "Wortbasiert::Zusätzliche Wörter": metrics["Zusätzliche Wörter"],
                    "Wortbasiert::WER": metrics["WER"],
                    "Wortbasiert::Word Error Rate inkl. fast korrekter Wörter": metrics["Word Error Rate inkl. fast korrekter Wörter"],
                    "Wortbasiert::Fast korrekt erkannte Wörter": metrics["Fast korrekt erkannte Wörter"],
                    "Wortbasiert::Korrekt erkannte Quote": metrics["Korrekt erkannte Quote"],
                    "Wortbasiert::Korrekt erkannte Quote inkl. fast korrekter Wörter": metrics["Korrekt erkannte Quote inkl. fast korrekter Wörter"],
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

    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Ergebnisse wurden in {excel_path} gespeichert.")

    summary_path = os.path.join(output_directory, "Summary_All.txt")
    details_path = os.path.join(output_directory, "Details_All.txt")

    with open(summary_path, 'w', encoding='utf-8') as summary_file:
        summary_file.writelines(summary_contents)

    with open(details_path, 'w', encoding='utf-8') as details_file:
        details_file.writelines(detailed_contents)

    print(f"Zusammenfassung und Details wurden gespeichert.")

# Beispielaufruf
process_all_files_to_excel(
    goldstandard_directory="../../Load Model Picture Input/Goldstandard/Infografiken",
    extracted_directory="../../Load Model Picture Input/Modell_Output/Llama/Diagramme und infografische Elemente",
    output_directory="Ergebnis_Llama",
    excel_path="Ergebnis_Llama/results.xlsx"
)
