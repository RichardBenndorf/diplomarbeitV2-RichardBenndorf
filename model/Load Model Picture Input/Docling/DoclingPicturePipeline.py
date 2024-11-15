import os
from docling.document_converter import DocumentConverter

# Verzeichnisse definieren
input_directory = "../Bilder/uneinheitliches Layout"
output_directory = "../Modell_Output/Docling/uneinheitliches Layout"

# Sicherstellen, dass das Output-Verzeichnis existiert
os.makedirs(output_directory, exist_ok=True)

# Dokumentenkonverter initialisieren
converter = DocumentConverter()

# Alle Bilddateien im Input-Verzeichnis durchlaufen
for filename in os.listdir(input_directory):
    # Nur Bilddateien verarbeiten (optional: andere Erweiterungen hinzufügen)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Pfade für Quelle und Ziel festlegen
        source_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
        
        # Bild konvertieren und Text extrahieren
        result = converter.convert(source_path)
        text = result.document.export_to_markdown()
        
        # Extrahierten Text in eine .txt-Datei schreiben
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)
        
        print(f"Text aus '{filename}' extrahiert und in '{output_path}' gespeichert.")
