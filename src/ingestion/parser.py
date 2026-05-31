import os
import fitz  # PyMuPDF
import docx

def read_txt(file_path: str) -> str:
    """Read plain text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(file_path: str) -> str:
    """Read and extract text from a PDF file using PyMuPDF."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"
    doc.close()
    return text

def read_docx(file_path: str) -> str:
    """Read and extract text from a Word Document."""
    doc = docx.Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def parse_document(file_path: str) -> str:
    """Parse document content based on its file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        return read_txt(file_path)
    elif ext == '.pdf':
        return read_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
