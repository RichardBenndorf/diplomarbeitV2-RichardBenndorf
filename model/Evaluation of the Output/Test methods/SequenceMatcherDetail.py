from difflib import SequenceMatcher

# Funktion zur Berechnung der Satzähnlichkeit mit SequenceMatcher
def sentence_similarity_detailed(sentence1, sentence2):
    matcher = SequenceMatcher(None, sentence1, sentence2)
    similarity = matcher.ratio()
    
    print(f"\nVergleich von Sätzen:\nModell-Ausgabe: {sentence1}\nReferenz: {sentence2}")
    print(f"Ähnlichkeit der Sätze: {similarity:.2f}")
    
    # Detaillierte Darstellung der Unterschiede
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            print(f"Ersetzung: '{sentence1[i1:i2]}' in der Modell-Ausgabe durch '{sentence2[j1:j2]}' in der Referenz")
        elif tag == 'delete':
            print(f"Löschung: '{sentence1[i1:i2]}' fehlt in der Referenz")
        elif tag == 'insert':
            print(f"Einfügung: '{sentence2[j1:j2]}' fehlt in der Modell-Ausgabe")
        elif tag == 'equal':
            print(f"Übereinstimmung: '{sentence1[i1:i2]}'")

    return similarity

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
total_similarity = 0  # Gesamte Ähnlichkeit für alle Zeilen

for i, (output_line, reference_line) in enumerate(zip(output_lines, reference_lines)):
    print(f"\nZeile {i+1}:")
    similarity = sentence_similarity_detailed(output_line, reference_line)
    total_similarity += similarity

# Berechne die Durchschnittsgenauigkeit
average_similarity = total_similarity / len(output_lines)
print(f"\nDurchschnittliche Genauigkeit für den gesamten Text: {average_similarity:.2f}")
