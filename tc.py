import spacy
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def create_structured_pdf(text, filename):
    # Load English tokenizer, tagger, parser, NER, and word vectors
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(text)

    # Initialize PDF document
    pdf = SimpleDocTemplate(f"{filename}.pdf", pagesize=letter)
    styles = getSampleStyleSheet()

    # Initialize content list
    content = []

    # Iterate over sentences in the text
    for sent in doc.sents:
        # Add sentence as paragraph to content
        content.append(Paragraph(sent.text, styles['Normal']))
        content.append(Spacer(1, 12))  # Add space after paragraph

    # Build PDF document
    pdf.build(content)

    return f"{filename}.pdf"

# Example text
example_text = """
Title: SRAM and DRAM: A Comprehensive Guide for Blind Children. Introduction: Hello! In this study material, we will explore the differences between Static Random Access Memory (SRAM) and Dynamic Random Access Memory (DRAM), two fundamental types of computer memory. We will present the information in an accessible way to help blind children understand these concepts more easily. Section 1: Understanding Computer Memory. Computer memory is a vital component that stores data and instructions for the computer to use. There are two primary types: Static Random Access Memory (SRAM) and Dynamic Random Access Memory (DRAM). Section 2: What is SRAM? SRAM, or Static Random Access Memory, retains data as long as power is supplied to the system. It utilizes flip-flops, which are bistable circuits capable of storing a single bit of data. Unlike DRAM, SRAM does not require constant refreshing to maintain data integrity. Commonly used in cache memory, where fast access to frequently accessed data is crucial for enhancing overall system performance. Has advantages such as speed and energy efficiency but is more expensive and less dense than DRAM. Section 3: What is DRAM? DRAM, or Dynamic Random Access Memory, stores data as electrical charges in capacitors which require periodic refreshing to prevent data loss. Each memory cell consists of a capacitor and a transistor, with the capacitor holding the data and the transistor acting as a switch to access it. Slower and consumes more power compared to SRAM but offers higher density and lower cost per bit, making it ideal for main memory in computer systems. Requires frequent refreshing which introduces latency and reduces overall system performance. Section 4: Key Differences between SRAM and DRAM. SRAM. DRAM. Retains data with power supply. Requires periodic refreshing. Faster access time. Slower access time. Energy-efficient. Less energy-efficient. More expensive and less dense. Cheaper and denser. Ideal for cache memory. Ideal for main memory. Section 5: Hands-on Activities. Matching Game: Identify which type of memory is described in a given sentence. Memory Cell Modeling: Build simple models using everyday items to understand how SRAM and DRAM cells function. Listening Quiz: Listen to a description of each memory type and then identify which one it corresponds to. Section 6: Conclusion. In this study material, we have explored the differences between SRAM and DRAM, two fundamental types of computer memory. We presented the information in a way that should be accessible to blind children to help them grasp these concepts more easily. Through hands-on activities and clear explanations, you now have the knowledge to understand how each type of memory is utilized in computer systems. Keep practicing, and you'll soon master the art of computer memory!
"""

# Example usage
output_pdf_path = create_structured_pdf(example_text, "structured_document")
print("PDF created at:", output_pdf_path)
