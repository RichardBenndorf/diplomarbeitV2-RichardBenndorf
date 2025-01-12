import os
from docling.document_converter import DocumentConverter

# Dateipfade definieren
image_path = "../Bilder/Kopf_Fußzeilen/Kopf_Fußzeilen_7.png"  # Pfad zur Bilddatei
output_file = "../Modell_Output/Docling/Kopf_Fußzeilen/Kopf_Fußzeilen_7.txt"  # Pfad zur Ausgabedatei

# Sicherstellen, dass der Zielordner existiert
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Dokumentenkonverter initialisieren
converter = DocumentConverter()

# Konvertiere das Bild und extrahiere den Text
result = converter.convert(image_path)
text = result.document.export_to_markdown()

# Speichere den extrahierten Text in einer .txt-Datei
with open(output_file, "w", encoding="utf-8") as file:
    file.write(text)

# Erfolgsmeldung ausgeben
print(f"Text aus '{image_path}' extrahiert und in '{output_file}' gespeichert.")
