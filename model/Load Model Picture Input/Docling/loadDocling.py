from docling.document_converter import DocumentConverter

source = "../testbilder/0.png"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())