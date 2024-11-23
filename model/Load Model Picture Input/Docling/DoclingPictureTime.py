import os
import time
from docling.document_converter import DocumentConverter

# Verzeichnisse definieren
input_directory = "../Bilder/uneinheitliches Layout"
output_directory = "../Modell_Output/Docling/uneinheitliches Layout"
durations_path = os.path.join(output_directory, "durations.txt")  # Pfad für die Zeitmessungsdatei

# Sicherstellen, dass das Output-Verzeichnis existiert
os.makedirs(output_directory, exist_ok=True)

# Dokumentenkonverter initialisieren
converter = DocumentConverter()

# Dummy-Bild laden, um das Modell zu initialisieren
dummy_image_path = "../DummyPicture/dummy.png"  # Stelle sicher, dass diese Bilddatei existiert
start_time = time.time()
converter.convert(dummy_image_path)  # Dummy-Konvertierung zur Initialisierung
model_load_duration = time.time() - start_time

# Datei für Zeitmessungen vorbereiten
with open(durations_path, "w", encoding="utf-8") as durations_file:
    durations_file.write(f"Modell Ladezeit, {model_load_duration:.2f} Sekunden\n")
    durations_file.write("Dateiname, Durchlaufzeit (Sekunden)\n")

# Alle Bilddateien im Input-Verzeichnis durchlaufen
for filename in os.listdir(input_directory):
    # Nur Bilddateien verarbeiten (optional: andere Erweiterungen hinzufügen)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Pfade für Quelle und Ziel festlegen
        source_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
        
        # Startzeit für die Bildverarbeitung messen
        start_time = time.time()
        
        # Bild konvertieren und Text extrahieren
        result = converter.convert(source_path)
        text = result.document.export_to_markdown()
        
        # Extrahierten Text in eine .txt-Datei schreiben
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)
        
        # Durchlaufzeit für die Bildverarbeitung berechnen
        duration = time.time() - start_time
        
        # Durchlaufzeit in die Datei schreiben
        with open(durations_path, "a", encoding="utf-8") as durations_file:
            durations_file.write(f"{filename}, {duration:.2f}\n")
        
        print(f"Text aus '{filename}' extrahiert und in '{output_path}' gespeichert. Durchlaufzeit: {duration:.2f} Sekunden")

print(f"Modell Ladezeit: {model_load_duration:.2f} Sekunden")
