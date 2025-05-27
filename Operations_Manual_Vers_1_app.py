import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_ollama import OllamaLLM
from langchain.docstore.document import Document # This import is still unused if not creating Document objects explicitly
from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks/Operational Rules and Procedures"

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "granite3-dense"

# -- organization names --
ORG_A_NAMES = ["ROPSSA", "Republic of Palau Social Security Administration", "SSA", 
               "Social Security Administration", "Social Security"]
ORG_B_NAMES = ["HCF", "Health Care Fund", "Republic of Palau Health Care Fund", 
               "Healthcare Fund", "ROPHCF"]

# -- list of directory paths to the individual files --
LIST_OF_MANUAL_FILES = [ "section 101-112.docx",
    "section 201-202.docx", "section 203–206.5.docx",
    "section 304.docx", "sections 206.5A–206.5B.docx",
    "sections 207–213.docx", "sections 214–215.docx",
    "sections 216–220.docx", "sections 305–317.docx",
    "sections 301–303.docx", "sections 318–325.docx",
    "sections 326–330.docx", "sections 401–407.docx",
    "sections 501–510.docx", "sections 601–603.docx",
    "sections 701–711.docx", "sections 801–807.docx",
    "sections 901–907.docx"
]

# --- initialization ---

print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.2) # initially, we're going to try temp of 0.2 reduce if needed

# --- Functions ---

# Load the entire content from a single chunk
# Returns full concatenated text and filename
def load_document_content(filepath):
    loader = None
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        print(f"Skipping unsupported file type: {filepath}")
        return None, None

    print(f"Loading raw content from {os.path.basename(filepath)}...")
    try:
        raw_docs = loader.load()
        # Store chunk content
        full_text = "\n\n".join([doc.page_content for doc in raw_docs])
        return full_text, os.path.basename(filepath)
    except Exception as e:
        print(f"Error loading {os.path.basename(filepath)}: {e}")
        return None, None
    
# Analyzes a single chunk to extract and separate content for two different orgs
def analyze_single_chunk(chunk_full_text, chunk_filename, org_a_names, org_b_names, llm_model):
    print(f"\n--- Analyzing Manual: '{chunk_filename}' for {org_a_names[0]} and {org_b_names[0]} ---")

    # Check if context text is too long for the LLM
    max_tokens = 4096 # Common context window size for many Ollama models
    # A typical token is about 4 characters. Adjust max_tokens based on your specific Ollama model's context window.

    if len(chunk_full_text) / 4 > max_tokens * 0.9: # Use 90% of max_tokens as a safer threshold
        print(f"WARNING: Manual '{chunk_filename}' content length ({len(chunk_full_text)} chars) might exceed LLM's context window ({max_tokens*4} chars est.).")
        print("         The LLM may only process a truncated portion of this manual.")

    org_a_all_names = ", ".join(name for name in org_a_names)
    org_b_all_names = ", ".join(name for name in org_b_names)
    org_a_name = org_a_names[0]
    org_b_name = org_b_names[0]

    prompt = f"""
    You are an expert operations manual analyst. You are provided with a chunk of text out of an operations manual.
    This manual contains information relevant to two distinct organizations: **{org_a_name}** and **{org_b_name}**.

    Note that {org_a_name} may be referred to as any of the following: {org_a_all_names}.
    And {org_b_name} may be reffered to as any of the following: {org_b_all_names}.

    Your task is to extract and present **ALL relevant information for {org_a_name} in PART 1** and **ALL relevant information for {org_b_name} in PART 2**.

    **IMPORTANT CONSIDERATION:**
    If a section or piece of information in the manual does not explicitly specify which organization it applies to (i.e., it doesn't mention "{org_a_name}" or any of its aliases, nor "{org_b_name}" or any of its aliases, and seems to be general policy), you must assume it applies to **BOTH** organizations. In such cases, include this general information verbatim under **PART 1: Information for {org_a_name}** AS WELL AS verbatim under **PART 2: Information for {org_b_name}**. This means content that applies to both organizatoins should appear **TWICE** in your response: once for PART 1 and once for PART 2.
    
    **YOUR OUTPUT MUST FOLLOW THIS STRICT STRUCTURE:**

    **PART 1: Information for {org_a_name}**
    [Insert all verbatim text from the manual that specifically applies to {org_a_name}, AS WELL AS general information applying to both.
    Maintain original section numbers and headings if present.]

    **PART 2: Information for {org_b_name}**
    [Insert all verbatim text from the manual that specifically applies to {org_b_name}, AS WELL AS general information applying to both.
    Maintain original section numbers and headings if present.]

    ---
    **Operations Manual Content (FOR YOUR REFERENCE AND VERBATIM COPY-PASTE):**
    ---
    {chunk_full_text}
    ---

    **Your Compiled Information (Start with PART 1, then PART 2):**
    """

    try:
        response = llm_model.invoke(prompt)
        return {"chunk_filename": chunk_filename, "llm_response": response} # Changed key back to chunk_filename
    except Exception as e:
        return {"chunk_filename": chunk_filename, "llm_response": f"Error processing '{chunk_filename}': {e}"} # Changed key back to chunk_filename

     

# --- Helper Function to write report to a PDF ---
def write_report_to_pdf(report_data, output_filepath):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_dir = "./fonts"
    regular_font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    italic_font_path = os.path.join(font_dir, "NotoSans-Italic.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    bolditalic_font_path = os.path.join(font_dir, "NotoSans-BoldItalic.ttf")

    if os.path.exists(regular_font_path):
        pdf.add_font("NotoSans", "", regular_font_path)
    else:
        print(f"Warning: NotoSans-Regular.ttf not found at {regular_font_path}. Using default font.")
    if os.path.exists(italic_font_path):
        pdf.add_font("NotoSans", "I", italic_font_path)
    if os.path.exists(bold_font_path):
        pdf.add_font("NotoSans", "B", bold_font_path)
    if os.path.exists(bolditalic_font_path):
        pdf.add_font("NotoSans", "BI", bolditalic_font_path)

    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 20)
    except:
        pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 12, "ROPSSA Operations Manuals - Extracted Organization Information", align='C')
    pdf.ln(15)

    for chunk_info in report_data: # Iterate over the list of results
        chunk_filename = chunk_info["chunk_filename"] # Accessing 'chunk_filename' here
        llm_response_content = chunk_info["llm_response"]

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Analysis for Manual: {chunk_filename}", 0, 'L')
        pdf.ln(4)

        try:
            pdf.set_font("NotoSans", "", 11)
        except:
            pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, llm_response_content.strip())
        pdf.ln(8)

        try:
            pdf.set_font("NotoSans", "I", 9)
        except:
            pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "-" * 100, align='C')
        pdf.ln(8)

    pdf.output(output_filepath)

def write_individual_response_to_pdf(llm_response_content, original_filename, output_filepath):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_dir = "./fonts"
    regular_font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    italic_font_path = os.path.join(font_dir, "NotoSans-Italic.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    bolditalic_font_path = os.path.join(font_dir, "NotoSans-BoldItalic.ttf")

    if os.path.exists(regular_font_path):
        pdf.add_font("NotoSans", "", regular_font_path)
    if os.path.exists(italic_font_path):
        pdf.add_font("NotoSans", "I", italic_font_path)
    if os.path.exists(bold_font_path):
        pdf.add_font("NotoSans", "B", bold_font_path)
    if os.path.exists(bolditalic_font_path):
        pdf.add_font("NotoSans", "BI", bolditalic_font_path)

    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 18) # Slightly smaller font for individual reports title
    except:
        pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Organization Information Extracted from: {original_filename}", align='C')
    pdf.ln(10)

    try:
        pdf.set_font("NotoSans", "", 10)
    except:
        pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, llm_response_content.strip())
    pdf.ln(5) 

    try:
        pdf.output(output_filepath)
    except Exception as e:
        print(f"Error saving individual PDF for {original_filename} to {output_filepath}: {e}")


# --- Main Program Logic ---
if __name__ == "__main__":
    if not os.path.exists(DOCUMENT_LIBRARY_PATH):
        os.makedirs(DOCUMENT_LIBRARY_PATH)
        print(f"Created directory: {DOCUMENT_LIBRARY_PATH}")
    
    font_dir = "./fonts"
    if not os.path.exists(font_dir):
        print(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.")
    else:
        required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
        for font_file in required_fonts:
            if not os.path.exists(os.path.join(font_dir, font_file)):
                print(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.")
    
    compiled_reports_list = []

    if not LIST_OF_MANUAL_FILES:
        print("Error: LIST_OF_MANUAL_FILES is empty. Please add file paths to process. Exiting.")
        exit()

    output_directory_base = "./OM_output_reports_version_1" # New output directory
    os.makedirs(output_directory_base, exist_ok=True)

    individual_reports_dir = os.path.join(output_directory_base, "individual_manual_reports")
    os.makedirs(individual_reports_dir, exist_ok=True)
    print(f"Individual manual reports will be saved in: {individual_reports_dir}")



    for chunk_filename_short in LIST_OF_MANUAL_FILES: # Iterating through your list of chunk filenames
        full_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, chunk_filename_short)
        
        chunk_full_text, loaded_filename = load_document_content(full_filepath) # Using chunk_full_text and loaded_filename

        if chunk_full_text is None:
            print(f"Skipping processing for {chunk_filename_short} due to loading error.")
            continue 

        print(f"Chunk {loaded_filename} loaded with {len(chunk_full_text)} characters.") # Using Chunk for consistency

        print(f"\n--- Starting Analysis for {loaded_filename} ---")

        analysis_result = analyze_single_chunk(chunk_full_text, loaded_filename, ORG_A_NAMES, ORG_B_NAMES, llm)

        compiled_reports_list.append(analysis_result)
        # save each individual LLM response to its own PDF ---
        # Create a clean base name from the original file (e.g., "my_manual" from "my_manual.pdf")
        base_name = os.path.splitext(loaded_filename)[0]
        individual_pdf_output_filename = os.path.join(individual_reports_dir, f"response_to_{base_name}.pdf")
        
        write_individual_response_to_pdf(
            analysis_result["llm_response"], 
            loaded_filename, # Pass the original filename for the PDF title
            individual_pdf_output_filename
        )
        print(f"Saved individual report for '{loaded_filename}' to: {individual_pdf_output_filename}")

        print(f"\n{'='*20} End of Analysis for '{loaded_filename}' {'='*20}\n")
    
    print("\n--- Full Compiled Report (Console Preview) ---")
    for chunk_report in compiled_reports_list: # Looping as chunk_report
        print(f"\n### Chunk: {chunk_report['chunk_filename']}\n") 
        print(chunk_report["llm_response"])
        print("\n--------------------------------------------------")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_filename = os.path.join(output_directory_base, f"consolidated_report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_reports_list, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")