import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # New import
from langchain_openai import ChatOpenAI
from fpdf import FPDF
from datetime import datetime

print("Starting Operations Manual Editor...") # Debugging message to indicate the script has started

TEMPERATURE = 0.4 # This controls the randomness of the LLM's responses. Lower values make it more deterministic and less creative.
# if you want more creative responses, you can increase this value (e.g., 0.5 or 0.7), but for compliance checks, a lower value is usually better.

# --- Configuration ---
# DOCUMENT_LIBRARY_PATH is now less critical as the target file is specified directly
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks" # Keep this for output paths if desired

# --- INITIALIZATION OF LLM ---
# Ensure the model is running in LM studio on your computer
lm_studio_base_url = "http://localhost:1234/v1" # Default LM Studio URL. Depending on your setup, you may need to change this. Check LM Studio settings.
print(f"Initializing LLM with LM Studio (local OpenAI-compatible API) from: {lm_studio_base_url}")
llm = ChatOpenAI(base_url=lm_studio_base_url, api_key="lm-studio", model="local-model", temperature=TEMPERATURE)

# -- Single Document to be processed --
# This is now a single file path, not a list.
MANUAL_FILE_TO_EDIT = "Copy of HCF Operations Manual FEED TO LLM.pdf"

CHUNK_SIZE = 4000 # Characters per chunk 
CHUNK_OVERLAP = 0 # No overlap between chunks as requested

# --- Functions ---

# Load the entire content from a single document.
# Now returns a list of LangChain Document objects (one per page or as loaded by loader)
def load_full_document_as_documents(filepath):
    loader = None
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        print(f"Skipping unsupported file type: {filepath}")
        return [], None

    print(f"Loading raw content from {os.path.basename(filepath)}...")
    try:
        raw_docs = loader.load()
        # Add source metadata to each loaded document for better traceability in chunks
        for doc in raw_docs:
            doc.metadata["source"] = os.path.basename(filepath)
        return raw_docs, os.path.basename(filepath)
    except Exception as e:
        print(f"Error loading {os.path.basename(filepath)}: {e}")
        return [], None

# --- analyze_single_chunk function remains largely the same ---
# It will now receive doc.page_content for chunk_full_text
def analyze_single_chunk(chunk_full_text, chunk_index, llm_model): # chunk_filename is now chunk_index for numbering
    print(f"\n--- Analyzing Chunk {chunk_index} ---")

    # Check if context text is too long for the LLM
    max_tokens = 4096 # Common context window size for many Ollama models
    # A typical token is about 4 characters. Adjust max_tokens based on your specific Ollama model's context window.

    if len(chunk_full_text) / 4 > max_tokens * 0.9: # Use 90% of max_tokens as a safer threshold
        print(f"WARNING: Chunk {chunk_index} content length ({len(chunk_full_text)} chars) might exceed LLM's context window ({max_tokens*4} chars est.).")
        print("         The LLM may only process a truncated portion of this chunk.")

    # --- PROMPT FOR EDITING AND CLARITY ---
    prompt = f"""
    You are an expert editor specializing in operations manuals and legal writings. Your task is to meticulously review and refine the provided text from an operations manual chunk.

    Your primary goal is to enhance clarity, correct grammar and typos, and improve wording, ensuring the original meaning and intent are perfectly preserved.

    **CRITICAL INSTRUCTIONS FOR EDITING AND CONTENT RETENTION:**
    1.  **Editing Scope**: The aim is to modernize, improve clarity, and fix grammatical errors including incorrect capitalization and typos while maintaining the original meaning. You must **NOT** change the substantive content or operational policies described in the text.
    2.  **No Summarization or Omission**: You must **NOT** summarize, paraphrase, or omit any substantive information, sections, or concepts from the original manual chunk. *IMPORTANT*: All original content that is relevant to operational policy *MUST* be present in your edited output.
    3.  **Title**: If section titles or headings are deemed inappropriate or unclear, you may suggest new titles that better reflect the content of the section. However, ensure that the original meaning is preserved.
    **Your output should ONLY be the polished, edited version of the operations manual content.** Do not add any introductory remarks, concluding statements, or meta-commentary from yourself.

    *** REMEMBER TO PRESERVE ALL CITATIONS!!!!!***

    ---
    **Operations Manual Content for Editing:**
    ---
    {chunk_full_text}
    ---

    **Your Polished Operations Manual Content:**
    """
    # --- END OF PROMPT ---

    try:
        response = llm_model.invoke(prompt)
        # Store a simple identifier for the chunk (e.g., "Chunk 1", "Chunk 2")
        return {"chunk_id": f"Chunk {chunk_index}", "llm_response": response}
    except Exception as e:
        return {"chunk_id": f"Chunk {chunk_index}", "llm_response": f"Error processing 'Chunk {chunk_index}': {e}"}

# --- Helper Functions to write report to a PDF (adjusted for new data structure) ---
def write_report_to_pdf(report_data, original_doc_name, output_filepath):
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
    pdf.multi_cell(0, 12, f"ROPSSA Operations Manual - Edited Content for: {original_doc_name}", align='C') # Changed title to reflect editing and original doc name
    pdf.ln(15)

    for chunk_info in report_data: # Iterate over the list of results
        chunk_id = chunk_info["chunk_id"] # Use the new chunk_id
        llm_response_content = chunk_info["llm_response"]

        # Handle potential error messages in llm_response
        if isinstance(llm_response_content, str):
            display_content = llm_response_content
        else:
            display_content = llm_response_content.content.strip()

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Edited Content for {chunk_id}", 0, 'L') # Changed title
        pdf.ln(4)

        try:
            pdf.set_font("NotoSans", "", 11)
        except:
            pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, display_content)
        pdf.ln(8)

        try:
            pdf.set_font("NotoSans", "I", 9)
        except:
            pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "-" * 100, align='C')
        pdf.ln(8)

    pdf.output(output_filepath)

def write_individual_response_to_pdf(llm_response_content, chunk_id, original_doc_name, output_filepath): # chunk_id replaces original_filename
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
    pdf.multi_cell(0, 10, f"Edited Content for {chunk_id} from {original_doc_name}", align='C') # Changed title
    pdf.ln(10)

    # Handle potential error messages in llm_response
    if isinstance(llm_response_content, str):
        display_content = llm_response_content
    else:
        display_content = llm_response_content.content.strip()

    try:
        pdf.set_font("NotoSans", "", 10)
    except:
        pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, display_content)
    pdf.ln(5)

    try:
        pdf.output(output_filepath)
    except Exception as e:
        print(f"Error saving individual PDF for {chunk_id} to {output_filepath}: {e}")

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

    full_filepath_to_edit = os.path.join(DOCUMENT_LIBRARY_PATH, MANUAL_FILE_TO_EDIT)

    if not os.path.exists(full_filepath_to_edit):
        print(f"Error: Manual file to edit '{MANUAL_FILE_TO_EDIT}' not found at '{DOCUMENT_LIBRARY_PATH}'. Please ensure it exists. Exiting.")
        exit()

    # Load the entire document as a list of LangChain Document objects
    # Renamed variable from full_document_text to raw_document_pages to reflect its content
    raw_document_pages, original_document_name = load_full_document_as_documents(full_filepath_to_edit)

    if not raw_document_pages: # Check if the list of documents is empty
        print(f"Error: No content loaded from '{MANUAL_FILE_TO_EDIT}'. Exiting.")
        exit()

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    # Split the loaded document pages into chunks
    document_chunks = text_splitter.split_documents(raw_document_pages)
    print(f"Document split into {len(document_chunks)} non-overlapping chunks using RecursiveCharacterTextSplitter.")

    output_directory_base = "./Manual_Editing_Results" # Base directory for all output files
    os.makedirs(output_directory_base, exist_ok=True)

    individual_reports_dir = os.path.join(output_directory_base, f"{os.path.splitext(original_document_name)[0]}_edited_chunks")
    os.makedirs(individual_reports_dir, exist_ok=True)
    print(f"Individual edited chunks will be saved in: {individual_reports_dir}")

    # Process each generated chunk
    for i, chunk_doc in enumerate(document_chunks): # Iterate over LangChain Document objects
        chunk_identifier = i + 1 # Start numbering from 1
        
        print(f"\n--- Starting Analysis for Chunk {chunk_identifier} ({len(chunk_doc.page_content)} characters) ---")

        # Pass the page_content of the chunk_doc to the analysis function
        analysis_result = analyze_single_chunk(chunk_doc.page_content, chunk_identifier, llm)
        compiled_reports_list.append(analysis_result)

        # Save individual edited chunk to PDF
        individual_pdf_output_filename = os.path.join(individual_reports_dir, f"edited_chunk_{chunk_identifier}.pdf")

        write_individual_response_to_pdf(
            analysis_result["llm_response"],
            analysis_result["chunk_id"], # Pass the formatted chunk_id
            original_document_name,
            individual_pdf_output_filename
        )
        print(f"Saved individual edited content for '{analysis_result['chunk_id']}' to: {individual_pdf_output_filename}")

        print(f"\n{'='*20} End of Analysis for {analysis_result['chunk_id']} {'='*20}\n")

    print("\n--- Full Compiled Edited Report (Console Preview) ---")
    for chunk_report in compiled_reports_list:
        print(f"\n### {chunk_report['chunk_id']}\n")
        # Handle potential error messages in llm_response
        if isinstance(chunk_report["llm_response"], str):
            print(chunk_report["llm_response"])
        else:
            print(chunk_report["llm_response"].content.strip())
        print("\n--------------------------------------------------")

    # Save the consolidated edited document report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_edited_{os.path.splitext(original_document_name)[0]}_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_reports_list, original_document_name, consolidated_pdf_filename)
        print(f"Successfully saved consolidated edited report to PDF: {consolidated_pdf_filename}")
    except Exception as e:
        print(f"Error saving consolidated PDF report: {e}")

    print("\nProgram Finished.")