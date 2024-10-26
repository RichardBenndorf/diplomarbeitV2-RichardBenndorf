import os
from difflib import SequenceMatcher

# Verzeichnispfade
output_folder = "../Load Model Picture Input/Modell_Output"
goldstandard_folder = "../Load Model Picture Input/Goldstandard"
evaluation_file = "Evaluation of the Output/evaluation_summary.txt"
detailed_output_folder = "Evaluation of the Output/Detailed_Comparison"

# Stelle sicher, dass der Ordner für die detaillierte Ausgabe existiert
os.makedirs(detailed_output_folder, exist_ok=True)

# Funktion zur Berechnung der besten Übereinstimmung zwischen zwei Zeilen
def best_match(line, reference_lines, threshold=0.5):
    best_similarity = 0
    best_ref_line = ""
    matcher = None
    for ref_line in reference_lines:
        temp_matcher = SequenceMatcher(None, line, ref_line)
        similarity = temp_matcher.ratio()
        if similarity > best_similarity:
            best_similarity = similarity
            best_ref_line = ref_line
            matcher = temp_matcher
    # Nur akzeptieren, wenn die Ähnlichkeit eine Mindestschwelle überschreitet
    if best_similarity >= threshold:
        return best_similarity, best_ref_line, matcher
    return 0, None, None

# Funktion zur Bereinigung von Text (entfernt nicht-wesentliche Zeichen)
def clean_text(text):
    return text.replace("**", "").replace("-", "").strip()

# Funktion zur detaillierten Darstellung der Unterschiede
def sentence_similarity_detailed(line, best_ref_line, matcher, file):
    similarity = matcher.ratio()
    
    file.write(f"\nVergleich von Sätzen:\nModell-Ausgabe: {line}\nReferenz: {best_ref_line}\n")
    file.write(f"Ähnlichkeit der Sätze: {similarity:.2f}\n")
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            file.write(f"Ersetzung: '{line[i1:i2]}' in der Modell-Ausgabe durch '{best_ref_line[j1:j2]}' in der Referenz\n")
        elif tag == 'delete':
            file.write(f"Löschung: '{line[i1:i2]}' fehlt in der Referenz\n")
        elif tag == 'insert':
            file.write(f"Einfügung: '{best_ref_line[j1:j2]}' fehlt in der Modell-Ausgabe\n")
        elif tag == 'equal':
            file.write(f"Übereinstimmung: '{line[i1:i2]}'\n")

    return similarity

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

            # Textzeilen aufteilen und bereinigen
            model_lines = [clean_text(line) for line in model_output.split('\n')]
            goldstandard_lines = [clean_text(line) for line in goldstandard.split('\n')]

            # Berechne die Satzähnlichkeit für jede Zeile und speichere detaillierte Unterschiede
            total_similarity = 0
            valid_line_count = 0  # Zählt die tatsächlich verglichenen Zeilen

            # Erstelle eine detaillierte Ausgabedatei für das Bild
            detailed_output_file = os.path.join(detailed_output_folder, f"{filename.replace('_output.txt', '_detailed.txt')}")
            with open(detailed_output_file, "w", encoding="utf-8") as detailed_file:
                used_ref_lines = []  # Verfolgung der verwendeten Referenzzeilen
                for model_line in model_lines:
                    # Finde die beste Übereinstimmung für die aktuelle Modellzeile
                    best_similarity, best_ref_line, matcher = best_match(model_line, [line for line in goldstandard_lines if line not in used_ref_lines])

                    if matcher is not None:
                        used_ref_lines.append(best_ref_line)  # Markiere die Referenzzeile als verwendet
                        similarity = sentence_similarity_detailed(model_line, best_ref_line, matcher, detailed_file)
                        total_similarity += similarity
                        valid_line_count += 1  # Zähle nur gültige Übereinstimmungen

            # Berechne die Durchschnittsgenauigkeit für dieses Bild
            avg_similarity = total_similarity / valid_line_count if valid_line_count > 0 else 0

            # Schreibe die Bewertung in der Evaluationsdatei
            eval_file.write(f"{filename.replace('_output.txt', '')}: Genauigkeit = {avg_similarity:.2f}\n")
            print(f"{filename.replace('_output.txt', '')}: Genauigkeit = {avg_similarity:.2f}")
