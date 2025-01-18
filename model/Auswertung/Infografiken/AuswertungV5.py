import os
from collections import Counter
import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics import edit_distance
from nltk.translate.bleu_score import sentence_bleu

def clean_text(text):
    return re.sub(r'[^\w\s]', '', text)

def cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = (tfidf_matrix[0] @ tfidf_matrix[1].T).toarray()[0][0]
    return cosine_sim

def find_near_matches(gold_words, extracted_words, threshold=2):
    near_matches = []
    gold_counter = Counter(gold_words)
    extracted_counter = Counter(extracted_words)

    for gold_word in list(gold_counter.keys()):
        for extracted_word in list(extracted_counter.keys()):
            if edit_distance(gold_word, extracted_word) <= threshold:
                match_count = min(gold_counter[gold_word], extracted_counter[extracted_word])
                near_matches.extend([(gold_word, extracted_word)] * match_count)
                gold_counter[gold_word] -= match_count
                extracted_counter[extracted_word] -= match_count

                if gold_counter[gold_word] <= 0:
                    del gold_counter[gold_word]
                if extracted_counter[extracted_word] <= 0:
                    del extracted_counter[extracted_word]
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

    unmatched_gold = [word for word in goldstandard_words if word not in extracted_words]
    unmatched_extracted = [word for word in extracted_words if word not in goldstandard_words]

    near_matches = find_near_matches(unmatched_gold, unmatched_extracted, threshold=2)
    fast_correct_missing = [match[0] for match in near_matches if match[0] in unmatched_gold]
    fast_correct_extra = [match[1] for match in near_matches if match[1] in unmatched_extracted]

    # Zeichenbasierte Metriken
    gold_chars = Counter(goldstandard)
    extracted_chars = Counter(extracted)

    char_correct = sum(min(gold_chars[char], extracted_chars[char]) for char in gold_chars)
    char_missing = sum(max(gold_chars[char] - extracted_chars.get(char, 0), 0) for char in gold_chars)
    char_extra = sum(max(extracted_chars[char] - gold_chars.get(char, 0), 0) for char in extracted_chars)
    cer = (char_missing + char_extra) / len(goldstandard) if len(goldstandard) > 0 else 0

    # Fehlende Wörter: Alle Wörter im Goldstandard, die nicht exakt im extrahierten Text enthalten sind
    word_missing = len(unmatched_gold)

    # Zusätzliche Wörter: Alle Wörter im extrahierten Text, die nicht exakt im Goldstandard enthalten sind
    word_extra = len(unmatched_extracted)

    # Korrekte Wörter: Alle Wörter, die exakt übereinstimmen
    word_correct = len([word for word in goldstandard_words if word in extracted_words])
    total_words = len(goldstandard_words)

    # WER und erweiterter WER
    wer = (word_missing + word_extra) / total_words
    extended_wer = (word_missing - len(fast_correct_missing) + word_extra - len(fast_correct_extra)) / total_words

    # Precision und erweiterte Precision
    precision = word_correct / (word_correct + word_extra) if word_correct + word_extra > 0 else 0
    extended_precision = word_correct / (word_correct + word_extra - len(fast_correct_extra)) if word_correct + word_extra - len(fast_correct_extra) > 0 else 0

    # Recall und erweiterter Recall
    recall = word_correct / (word_correct + word_missing) if word_correct + word_missing > 0 else 0
    extended_recall = word_correct / (word_correct + word_missing - len(fast_correct_missing)) if word_correct + word_missing - len(fast_correct_missing) > 0 else 0

    # F1-Score und erweiterter F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    extended_f1 = 2 * (extended_precision * extended_recall) / (extended_precision + extended_recall) if extended_precision + extended_recall > 0 else 0

    # Semantische Metriken
    cosine_sim = cosine_similarity(goldstandard, extracted)
    bleu_score = sentence_bleu([goldstandard_words], extracted_words)

    # Detaillierte Vergleichsliste erstellen
    detailed_comparison = []
    for word in goldstandard_words:
        if word in extracted_words:
            detailed_comparison.append(f"KORREKT: {word}")
        elif word in fast_correct_missing:
            detailed_comparison.append(f"FAST KORREKT FEHLEND: {word}")
        else:
            detailed_comparison.append(f"FEHLT: {word}")

    for word in extracted_words:
        if word not in goldstandard_words and word not in fast_correct_extra:
            detailed_comparison.append(f"ZUSÄTZLICH: {word}")
        elif word in fast_correct_extra:
            detailed_comparison.append(f"FAST KORREKT ZUSÄTZLICH: {word}")

    detailed_content = "\n".join(detailed_comparison)

    # Zusammenfassung erstellen
    summary_content = f"""
Zusammenfassung der Metriken:
--------------------------------
Zeichenbasierte Metriken:
- Gesamtzeichen: {len(goldstandard)}
- Korrekte Zeichen: {char_correct}
- Fehlende Zeichen: {char_missing}
- Zusätzliche Zeichen: {char_extra}
- Character Error Rate (CER): {cer:.4f}

Wortbasierte Metriken:
- Gesamtwörter: {total_words}
- Korrekt erkannte Wörter: {word_correct}
- Fehlende Wörter: {word_missing}
- Zusätzliche Wörter: {word_extra}
- Fast korrekt fehlend: {len(fast_correct_missing)}
- Fast korrekt zusätzlich: {len(fast_correct_extra)}
- WER: {wer:.4f}
- Erweiterter WER: {extended_wer:.4f}

Semantische Metriken:
- Cosine Similarity: {cosine_sim:.4f}
- BLEU Score: {bleu_score:.4f}

Weitere Metriken:
- Precision: {precision:.4f}
- Precision erweitert: {extended_precision:.4f}
- Recall: {recall:.4f}
- Recall erweitert: {extended_recall:.4f}
- F1-Score: {f1:.4f}
- F1-Score erweitert: {extended_f1:.4f}
"""

    return summary_content, detailed_content, {
        "Gesamtzeichen": len(goldstandard),
        "Korrekte Zeichen": char_correct,
        "Fehlende Zeichen": char_missing,
        "Zusätzliche Zeichen": char_extra,
        "CER": cer,
        "Gesamtwörter": total_words,
        "Korrekt erkannte Wörter": word_correct,
        "Fehlende Wörter": word_missing,
        "Zusätzliche Wörter": word_extra,
        "Fast korrekt fehlend": len(fast_correct_missing),
        "Fast korrekt zusätzlich": len(fast_correct_extra),
        "WER": wer,
        "Erweiterter WER": extended_wer,
        "Precision": precision,
        "Precision erweitert": extended_precision,
        "Recall": recall,
        "Recall erweitert": extended_recall,
        "F1-Score": f1,
        "F1-Score erweitert": extended_f1,
        "Cosine Similarity": cosine_sim,
        "BLEU Score": bleu_score
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
            extracted_path = os.path.join(extracted_directory, f"Diagramme_und_infografische_Elemente_{index}.txt")

            if os.path.exists(extracted_path):
                summary, details, metrics = evaluate_extraction(goldstandard_path, extracted_path)

                metrics["Durchlauf"] = index
                results.append(metrics)

                summary_contents.append(f"Durchlauf {index}:{summary}\n")
                detailed_contents.append(f"Durchlauf {index}:{details}\n")

    # Ergebnisse in Excel speichern
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False, float_format="%.4f")

    # Zusammenfassung und Details speichern
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
    extracted_directory="../../Load Model Picture Input/Modell_Output/Molmo/Diagramme und infografische Elemente",
    output_directory="Ergebnis_Molmo",
    excel_path="Ergebnis_Molmo/results.xlsx"
)
