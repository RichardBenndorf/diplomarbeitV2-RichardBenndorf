from difflib import SequenceMatcher

# Funktion zur Bestimmung der Ähnlichkeit von Sätzen (Sentence Similarity)
def sentence_similarity(sentence1, sentence2):
    return SequenceMatcher(None, sentence1, sentence2).ratio()

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

# Vergleich der einzelnen Zeilen/Sätze
for i, (output_line, reference_line) in enumerate(zip(output_lines, reference_lines)):
    # Satzähnlichkeit prüfen
    sentence_similarity_score = sentence_similarity(output_line, reference_line)

    print(f"Zeile {i+1}:")
    print(f"Modell-Ausgabe: {output_line}")
    print(f"Referenz: {reference_line}")
    print(f"Ähnlichkeit der Sätze: {sentence_similarity_score:.2f}\n")
