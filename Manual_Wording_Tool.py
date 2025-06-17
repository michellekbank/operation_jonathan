import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import ChatOpenAI
from fpdf import FPDF
from datetime import datetime

print("Starting ROPSSA Operations Manual Editor...") # Debugging message to indicate the script has started

TEMPERATURE = 0.2 # This controls the randomness of the LLM's responses. Lower values make it more deterministic and less creative.
# if you want more creative responses, you can increase this value (e.g., 0.5 or 0.7), but for compliance checks, a lower value is usually better.
# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks/Operational Rules and Procedures"

# --- INITIALIZATION OF LLM ---

# Ensure the model is running in LM studio on your computer
lm_studio_base_url = "http://localhost:1234/v1"  # Default LM Studio URL. Depending on your setup, you may need to change this. Check LM Studio settings.
print(f"Initializing LLM with LM Studio (local OpenAI-compatible API) from: {lm_studio_base_url}")
llm = ChatOpenAI(base_url=lm_studio_base_url, api_key="lm-studio", model="local-model", temperature=TEMPERATURE)


# -- list of directory paths to the individual files --
LIST_OF_MANUAL_FILES = [#"section 101-112.docx",
                         #"section 201-202.docx", "section 203–204.docx", 
                         "section 205–206.5.docx", "section 304.docx", 
                         "section 323-325.docx", "section 506-510.docx", 
                         "section 707-711.docx", "sections 206.5A–206.5B.docx", 
                         "sections 207–213.docx", "sections 214–215.docx", 
                         "sections 216–218.docx", "sections 219–220.docx", 
                         "sections 301–303.docx", "sections 305–309.docx", 
                         "sections 310–317.docx", "sections 318–322.docx", 
                         "sections 326–330.docx", "sections 401–407.docx", 
                         "sections 501–505.docx", "sections 601–603.docx", 
                         "sections 701–706.docx", "sections 801–807.docx", 
                         "sections 901–907.docx"
]


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
    
# Analyzes a single chunk (now performs editing)
def analyze_single_chunk(chunk_full_text, chunk_filename, llm_model):
    print(f"\n--- Analyzing Manual: '{chunk_filename}' ---")

    # Check if context text is too long for the LLM
    max_tokens = 4096 # Common context window size for many Ollama models
    # A typical token is about 4 characters. Adjust max_tokens based on your specific Ollama model's context window.

    if len(chunk_full_text) / 4 > max_tokens * 0.9: # Use 90% of max_tokens as a safer threshold
        print(f"WARNING: Manual '{chunk_filename}' content length ({len(chunk_full_text)} chars) might exceed LLM's context window ({max_tokens*4} chars est.).")
        print("         The LLM may only process a truncated portion of this manual.")

    # --- NEW PROMPT FOR EDITING AND CLARITY ---
    prompt = f"""
    You are an expert editor specializing in operations manuals and legal writings. Your task is to meticulously review and refine the provided text from an operations manual chunk.

    *** IMPORTANT! YOU MUST PRESERVE CITATIONS FOUND IN THE TEXT!*** Without this, the response is USELESS. Every citation of the form [41 PNC §#] should be preserved EXACTLY AS IS in as close to the same location as possible

    Your primary goal is to enhance the clarity, correct grammar and typos, and improve wording, ensuring the original meaning and intent are perfectly preserved.

    **CRITICAL INSTRUCTIONS FOR EDITING AND CONTENT RETENTION:**
    1.  **Editing Scope**: The aim is to make the text as easy to understand as possible without altering its factual basis or the original meaning.
    2.  **No Summarization or Omission**: You must **NOT** summarize, paraphrase, or omit any substantive information, sections, or concepts from the original manual chunk. *IMPORTANT*: All original content that is relevant to operational policy *MUST* be present in your edited output.
    3.  **Title**: If section titles or headings are deemed inappropriate or unclear, you may suggest new titles that better reflect the content of the section. However, ensure that the original meaning is preserved.
    **Your output should ONLY be the polished, edited version of the operations manual content.** Do not add any introductory remarks, concluding statements, or meta-commentary from yourself.

    *** REMEMBER TO PRESERVE ALL CITATIONS!!!!!***

    ---
    **Operations Manual Content for Editing:**
    ---
    {chunk_full_text}
    ---

    **Your Polished Operations Manual Content:**
    """
    # --- END OF NEW PROMPT ---

    try:
        response = llm_model.invoke(prompt)
        return {"chunk_filename": chunk_filename, "llm_response": response} 
    except Exception as e:
        return {"chunk_filename": chunk_filename, "llm_response": f"Error processing '{chunk_filename}': {e}"}

     

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
        pdf.add_font("NotoSans", "", regular_font_path, uni=True) # Ensure unicode support
    else:
        print(f"Warning: NotoSans-Regular.ttf not found at {regular_font_path}. Using default font.")
    if os.path.exists(italic_font_path):
        pdf.add_font("NotoSans", "I", italic_font_path, uni=True)
    if os.path.exists(bold_font_path):
        pdf.add_font("NotoSans", "B", bold_font_path, uni=True)
    if os.path.exists(bolditalic_font_path):
        pdf.add_font("NotoSans", "BI", bolditalic_font_path, uni=True)

    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 20)
    except:
        pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 12, "ROPSSA Operations Manuals - Edited Content", align='C') # Changed title to reflect editing
    pdf.ln(15)

    for chunk_info in report_data: # Iterate over the list of results
        chunk_filename = chunk_info["chunk_filename"] 
        llm_response_content = chunk_info["llm_response"]

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Edited Content for Manual Chunk: {chunk_filename}", 0, 'L') # Changed title
        pdf.ln(4)

        try:
            pdf.set_font("NotoSans", "", 11)
        except:
            pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, llm_response_content.content.strip())
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
        pdf.add_font("NotoSans", "", regular_font_path, uni=True)
    if os.path.exists(italic_font_path):
        pdf.add_font("NotoSans", "I", italic_font_path, uni=True)
    if os.path.exists(bold_font_path):
        pdf.add_font("NotoSans", "B", bold_font_path, uni=True)
    if os.path.exists(bolditalic_font_path):
        pdf.add_font("NotoSans", "BI", bolditalic_font_path, uni=True)

    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 18) # Slightly smaller font for individual reports title
    except:
        pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Edited Content Extracted from: {original_filename}", align='C') # Changed title to reflect editing
    pdf.ln(10)

    try:
        pdf.set_font("NotoSans", "", 10)
    except:
        pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, llm_response_content.content.strip())
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

    output_directory_base = "./Manual_Word_Editing_Results"
    os.makedirs(output_directory_base, exist_ok=True)

    individual_reports_dir = os.path.join(output_directory_base, "manual_word_edits")
    os.makedirs(individual_reports_dir, exist_ok=True)
    print(f"Individual manual reports will be saved in: {individual_reports_dir}")


    for chunk_filename_short in LIST_OF_MANUAL_FILES:
        full_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, chunk_filename_short)
        
        chunk_full_text, loaded_filename = load_document_content(full_filepath)

        if chunk_full_text is None:
            print(f"Skipping processing for {chunk_filename_short} due to loading error.")
            continue 

        print(f"Chunk {loaded_filename} loaded with {len(chunk_full_text)} characters.")

        print(f"\n--- Starting Analysis for {loaded_filename} ---")

        analysis_result = analyze_single_chunk(chunk_full_text, loaded_filename, llm)

        compiled_reports_list.append(analysis_result)
        
        base_name = os.path.splitext(loaded_filename)[0]
        individual_pdf_output_filename = os.path.join(individual_reports_dir, f"response_to_{base_name}.pdf")
        
        write_individual_response_to_pdf(
            analysis_result["llm_response"], 
            loaded_filename, 
            individual_pdf_output_filename
        )
        print(f"Saved individual report for '{loaded_filename}' to: {individual_pdf_output_filename}")

        print(f"\n{'='*20} End of Analysis for '{loaded_filename}' {'='*20}\n")
    
    print("\n--- Full Compiled Report (Console Preview) ---")
    for chunk_report in compiled_reports_list:
        print(f"\n### Chunk: {chunk_report['chunk_filename']}\n")
        print(chunk_report["llm_response"])
        print("\n--------------------------------------------------")

    consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    try:
        write_report_to_pdf(compiled_reports_list, consolidated_pdf_filename)
        print(f"Successfully saved consolidated report to PDF: {consolidated_pdf_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")