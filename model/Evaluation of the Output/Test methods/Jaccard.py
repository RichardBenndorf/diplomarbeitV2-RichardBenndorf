import string
from difflib import SequenceMatcher

# Funktion zur Berechnung der Jaccard-Ähnlichkeit
def jaccard_similarity(word1, word2):
    set1 = set(word1)
    set2 = set(word2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Funktion zum flexiblen Vergleich der Wortpaare mit Fehlerbewertung
def compare_words_jaccard_with_transposition_and_error_scores(output_words, reference_words):
    total_error_score = 0  # Gesamter Fehlerwert
    i = 0  # Index für die Ausgabe-Wörter
    j = 0  # Index für die Referenz-Wörter

    while i < len(output_words) and j < len(reference_words):
        # Prüfe, ob Wörter gleich sind
        if output_words[i] == reference_words[j]:
            print(f"  Wort: '{output_words[i]}' und '{reference_words[j]}' sind gleich.")
            error_score = 0  # Keine Fehler
            i += 1
            j += 1
        # Prüfe auf Transposition (Wortvertauschung)
        elif i + 1 < len(output_words) and j + 1 < len(reference_words) and output_words[i] == reference_words[j + 1] and output_words[i + 1] == reference_words[j]:
            print(f"  Vertauschung erkannt: '{output_words[i]}' und '{output_words[i+1]}'")
            error_score = 0.5  # Fehlerwert für Transposition
            i += 2
            j += 2
        else:
            # Wörter unterscheiden sich, berechne Jaccard-Ähnlichkeit
            similarity = jaccard_similarity(output_words[i], reference_words[j])
            error_score = 1 - similarity  # Fehlerwert basiert auf Jaccard-Ähnlichkeit
            print(f"  Wort: '{output_words[i]}' vs. '{reference_words[j]}' - Jaccard-Ähnlichkeit: {similarity:.2f}, Fehlerwert: {error_score:.2f}")
            i += 1
            j += 1

        total_error_score += error_score

    # Überprüfe, ob Wörter übrig sind (Einfügungen oder Löschungen)
    while i < len(output_words):
        print(f"  Löschung: '{output_words[i]}' fehlt in der Referenz.")
        error_score = 1  # Fehlerwert für fehlendes Wort
        total_error_score += error_score
        i += 1
    while j < len(reference_words):
        print(f"  Einfügung: '{reference_words[j]}' fehlt in der Ausgabe.")
        error_score = 1  # Fehlerwert für zusätzliches Wort
        total_error_score += error_score
        j += 1

    # Berechnung des relativen Fehlerwerts
    total_words = max(len(output_words), len(reference_words))  # Maximale Anzahl der Wörter (um nicht zu unterbewerten)
    relative_error_score = total_error_score / total_words if total_words != 0 else 0  # Vermeide Division durch 0

    return total_error_score, relative_error_score

# Beispielausgabe (vom Modell generiert)
output_text = """
1 DE NATA COCO 0
1 COOKIE DOH SAUCES 0
1 NATA DE COCO 0
Sub Total 45,455
PB1 (10%) 4,545
Rounding 0
Total 50,000

Card Payment 50,000
"""

# Manuelle Referenz
reference_text = """
1 NATA DE COCO 0
1 COOKIE DOUGH SAUCES 0
1 NATA DE COCO 0
Subtotal 45,455
PB1 (10%) 4,545
Rounding 0
Total 50,000

Card Payment 50,000
"""

# Text in Sätze/Tokens aufteilen
output_lines = output_text.strip().split('\n')
reference_lines = reference_text.strip().split('\n')

# Vergleich der einzelnen Zeilen/Wörter
total_error = 0  # Gesamter Fehlerwert für alle Zeilen
total_relative_error = 0  # Gesamter relativer Fehlerwert für alle Zeilen

for i, (output_line, reference_line) in enumerate(zip(output_lines, reference_lines)):
    # Token-Wise Vergleich für flexiblen Wortvergleich
    output_words = output_line.split()
    reference_words = reference_line.split()

    print(f"Zeile {i+1}:")
    print(f"Modell-Ausgabe: {output_line}")
    print(f"Referenz: {reference_line}")

    # Flexibler Vergleich der Wörter mit Jaccard-Ähnlichkeit, Transpositionserkennung und Fehlerbewertung
    error_score, relative_error_score = compare_words_jaccard_with_transposition_and_error_scores(output_words, reference_words)
    total_error += error_score
    total_relative_error += relative_error_score
        
    print(f"Fehlerwert für Zeile {i+1}: {error_score:.2f}")
    print(f"Relativer Fehlerwert für Zeile {i+1}: {relative_error_score:.2f}\n")

# Ausgabe des gesamten Fehlerwerts
print(f"Gesamter Fehlerwert für den gesamten Text: {total_error:.2f}")
print(f"Gesamter relativer Fehlerwert für den gesamten Text: {total_relative_error:.2f}")
