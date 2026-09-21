"""
Create a sample PDF file for testing the RAG chatbot
Run this script to generate sample_data.pdf in your data/ folder
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

def create_sample_pdf():
    """Create a sample PDF with some test content."""
    
    # Make sure data folder exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    pdf_path = data_dir / "sample_data.pdf"
    
    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "Sample Document for RAG Testing")
    
    # Content
    c.setFont("Helvetica", 12)
    y = height - 100
    line_height = 20
    
    content = [
        "This is a sample PDF document for testing your RAG chatbot.",
        "",
        "About Python:",
        "Python is a high-level, interpreted programming language created by Guido van Rossum.",
        "It was first released in 1991 and emphasizes code readability with significant whitespace.",
        "Python is known for its simplicity and versatility in various domains including web",
        "development, data science, artificial intelligence, and automation.",
        "",
        "Key Features of Python:",
        "1. Easy to learn and read syntax",
        "2. Dynamically typed language",
        "3. Large standard library",
        "4. Cross-platform compatibility",
        "5. Strong community support",
        "",
        "Python Applications:",
        "- Web Development (Django, Flask)",
        "- Data Analysis (Pandas, NumPy)",
        "- Machine Learning (TensorFlow, PyTorch)",
        "- Automation and Scripting",
        "- Scientific Computing",
        "",
        "Machine Learning Basics:",
        "Machine learning is a subset of artificial intelligence that enables systems to learn",
        "and improve from experience without being explicitly programmed. There are three main",
        "types of machine learning: supervised learning, unsupervised learning, and",
        "reinforcement learning.",
        "",
        "Supervised learning uses labeled training data to make predictions. Examples include",
        "classification and regression tasks. Unsupervised learning finds patterns in unlabeled",
        "data, such as clustering and dimensionality reduction.",
        "",
        "Natural Language Processing:",
        "Natural Language Processing (NLP) is a branch of AI that focuses on enabling computers",
        "to understand, interpret, and generate human language. Common NLP tasks include text",
        "classification, sentiment analysis, named entity recognition, and machine translation.",
        "",
        "NLP powers many modern applications such as chatbots, voice assistants, and search",
        "engines. Recent advances in deep learning and transformer models like BERT and GPT",
        "have significantly improved NLP capabilities.",
    ]
    
    for line in content:
        if y < 50:  # Create new page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
        
        c.drawString(50, y, line)
        y -= line_height
    
    # Save PDF
    c.save()
    print(f"✅ Sample PDF created: {pdf_path}")
    print(f"   Location: {pdf_path.absolute()}")
    print(f"\nNow run: python ingest_ollama.py")

if __name__ == "__main__":
    create_sample_pdf()