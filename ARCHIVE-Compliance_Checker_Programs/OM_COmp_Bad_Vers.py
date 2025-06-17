import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaLLM 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document 

from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks/Operational Rules and Procedures"

GUIDELINES_DOC_NAME = "41 PNCA 2025.pdf" # Name of your guidelines document

# -- organization names (used in prompts for context, not for splitting output) --
ORG_A_NAMES = ["ROPSSA", "Republic of Palau Social Security Administration", "SSA", 
               "Social Security Administration", "Social Security"]
ORG_B_NAMES = ["HCF", "Health Care Fund", "Republic of Palau Health Care Fund", 
               "Healthcare Fund", "ROPHCF"]

# -- list of directory paths to the individual files (your manual chunks) --
LIST_OF_MANUAL_FILES = ["section 101-112.docx",
                         "section 201-202.docx", "section 203–204.docx", 
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


# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "mistral:7b-instruct-q4_k_m"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" 
TOP_K_GUIDELINES = 7 # Number of top relevant guideline chunks to retrieve for each aspect

# Define specific aspects/questions for targeted compliance checks
# IMPORTANT: Replace this with actual aspects you want to check!
COMPLIANCE_ASPECTS_TO_CHECK = [
   "Functions, members, and procedures of the Social Security Board",
    "Actuarial Soundness and Sustainability",
    "Fund Reserves and Solvency Requirements",
    "Audit Requirements and External Oversight",
    "Duties, functions, appointment of the Social Security Administrator",
    "Secretaries, managers, and other staff",
    "Financial reporting and budget",
    "Governance Structure and Oversight Mechanisms (including the National Healthcare Financing Governing Committee or the “Committee”)",
    "Enrollment and eligibility criteria",
    "Data management, security, and information sharing mechanisms and policies",
    "Appeals and Dispute Resolution Mechanisms",
    "Beneficiary Rights and Responsibilities",
    "Investment Policies, Portfolio Management, and Performance Reporting",
    "Incomes and contributions or payments",
    "Claims",
    "Aspects of health insurance, including benefits, exclusions, reimbursements, and subscriptions",
    "Privacy",
    "Employee offenses and penalties including fraud, failure to report or pay, false claims",
    "Enforcement Powers and Sanctions for Non-Compliance (beyond just offenses)",
    "Succession and transfer of medical savings account after death",
    "The keeping of accounts and reports"
]

# --- INITIALIZATION ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.2) 

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# --- FUNCTIONS ---

def load_document_content(filepath):
    """
    Loads the entire text content from a single document.
    Returns a list of LangChain Document objects (one per page) and the filename.
    """
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
        # Add source metadata to each loaded document
        for doc in raw_docs:
            doc.metadata["source"] = os.path.basename(filepath)
        return raw_docs, os.path.basename(filepath)
    except Exception as e:
        print(f"Error loading {os.path.basename(filepath)}: {e}")
        return [], None

def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits a list of LangChain Document objects into smaller chunks.
    This is used for the guidelines document to create a vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks, embeddings_model):
    """Creates a FAISS vector store from document chunks and embeddings."""
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

def get_relevant_guideline_context(query_aspect, guidelines_vectorstore, top_k_value):
    """
    Helper function to retrieve and format relevant guideline chunks.
    Returns a formatted string of guidelines and a boolean indicating if any were found.
    """
    relevant_guidelines_docs = guidelines_vectorstore.similarity_search(query_aspect, k=top_k_value) 
    
    if not relevant_guidelines_docs:
        return "", False

    guidelines_context_parts = []
    for i, doc in enumerate(relevant_guidelines_docs):
        guidelines_context_parts.append(
            f"Guideline Excerpt {i+1} (Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}):\n"
            f"\"{doc.page_content}\""
        )
    guidelines_context = "\n\n".join(guidelines_context_parts)
    return guidelines_context, True


def check_manual_compliance(manual_chunk_full_text, manual_chunk_filename, guidelines_vectorstore, llm_model, compliance_aspects, top_k_value):
    """
    Checks the compliance of a single manual chunk against guidelines from a vector store,
    focusing on specific compliance aspects.
    """
    print(f"\n--- Checking Compliance for Manual Chunk: '{manual_chunk_filename}' ---")
    compliance_report_content = [] 

    # Iterating through each compliance aspect for a targeted check
    if not compliance_aspects:
        compliance_report_content.append("No specific compliance aspects defined. Cannot perform targeted checks.")
        print("No specific compliance aspects defined.")
        return "\n".join(compliance_report_content)

    for aspect in compliance_aspects:
        print(f"  Checking aspect: '{aspect}'...")

        relevance_prompt = f"""
        Given the following Operations Manual Chunk and the Compliance Aspect: "{aspect}",
        does the Manual Chunk contain any information or discussion directly relevant to this aspect?
        Respond only with "YES", "NO", or "AMBIGUOUS".

        Operations Manual Chunk:
        ---
        {manual_chunk_full_text}
        ---
        Compliance Aspect: "{aspect}"
        """
        relevance_response = llm_model.invoke(relevance_prompt).strip().upper()

        if relevance_response == "NO":
            compliance_report_content.append(
                f"\n**Compliance Aspect: {aspect}**\n"
                f"  - Status: NOT ADDRESSED (Manual chunk contains no relevant information for this aspect)."
            )
            continue # Skip the main compliance prompt for this aspect
        
        # Use the helper function to get relevant guidelines
        guidelines_context, guidelines_found = get_relevant_guideline_context(aspect, guidelines_vectorstore, top_k_value)
        
        if not guidelines_found:
            compliance_report_content.append(
                f"\n**Compliance Aspect: {aspect}**\n"
                f"  - Status: NOT APPLICABLE (No relevant guidelines found for this aspect)."
            )
            continue

        # --- LLM Prompt for Compliance Check (MODIFIED for clarity in status) ---
        prompt = f"""
        You are a highly analytical compliance officer. Your task is to meticulously evaluate whether the
        'Operations Manual Chunk' explicitly complies with the 'Relevant Guidelines' provided,
        specifically focusing on the aspect: "{aspect}".

        **CRITICAL INSTRUCTIONS FOR ANALYSIS AND REPORTING:**
        1.  **Compliance Status:** Begin your response for this aspect with one of the following, choosing the most appropriate based on the *presence and content* of the manual chunk:
            * **COMPLIANT:** If the manual chunk fully and clearly meets the guideline.
            * **NON-COMPLIANT:** If the manual chunk clearly violates, contradicts, or presents a significant deviation from the guideline.
            * **PARTIALLY COMPLIANT:** If the manual chunk *attempts to address* the guideline but does so incompletely, vaguely, or with minor deficiencies that prevent full adherence.
            * **NOT ADDRESSED:** If the 'Operations Manual Chunk' **does not contain sufficient information or discussion relevant to this specific guideline aspect**, or if the aspect is entirely absent from the manual's content. Do not use 'PARTIALLY COMPLIANT' if the manual simply lacks content on the topic.

        2.  **Explanation & Reasoning:** Provide a concise, objective explanation for your status determination.
        3.  **Verbatim Citations (Crucial):** You MUST cite specific, verbatim phrases or sentences from **both** the 'Operations Manual Chunk' and the 'Relevant Guidelines' to support your reasoning. For each citation, include its original source (e.g., "Manual: '...' (from Section X.Y)", "Guideline: '...' (from Page Z)"). These citations are paramount for traceability. If no direct citation from the manual can be found to support a compliance claim for 'COMPLIANT' or 'NON-COMPLIANT', consider if it is 'NOT ADDRESSED'.

        **Operations Manual Chunk for Evaluation (from {manual_chunk_filename}):**
        ---
        {manual_chunk_full_text}
        ---

        **Relevant Guidelines (related to '{aspect}'):**
        ---
        {guidelines_context}
        ---

        **Compliance Analysis for Aspect: "{aspect}"**
        """
        # --- End of LLM Prompt ---

        try:
            response = llm_model.invoke(prompt)
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n{response.strip()}")
        except Exception as e:
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n  - Error during analysis: {e}")
            print(f"  Error analyzing aspect '{aspect}' for '{manual_chunk_filename}': {e}")
            
    return "\n".join(compliance_report_content)

# --- Helper Function to write the consolidated report to a PDF (MODIFIED) ---
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
    pdf.multi_cell(0, 12, "ROPSSA Operations Manuals - Compliance Report", align='C') 
    pdf.ln(15)

    for chunk_info in report_data: 
        chunk_filename = chunk_info["chunk_filename"] 
        llm_response_content = chunk_info["llm_response"]

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Compliance Analysis for Manual Chunk: {chunk_filename}", 0, 'L') 
        pdf.ln(4)

        # Split the LLM response by aspect headers to apply formatting
        # This regex looks for lines starting with "\n**Compliance Aspect: " followed by text and then "**\n"
        # We use re.split and re.findall to handle the separation while keeping the delimiters.
        aspect_sections = re.split(r'(\n\*\*Compliance Aspect:.*? \*\*)\n', llm_response_content, flags=re.DOTALL)
        
        # The first part might be empty if the report starts with an aspect header
        if aspect_sections and aspect_sections[0].strip():
            try:
                pdf.set_font("NotoSans", "", 11)
            except:
                pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, aspect_sections[0].strip())

        for i in range(1, len(aspect_sections), 2): # Iterate over pairs of (header, content)
            header = aspect_sections[i].strip()
            content = aspect_sections[i+1].strip() if i+1 < len(aspect_sections) else ""

            # Apply bold and slightly larger font for the aspect header
            pdf.ln(4) # Add a small line break before the header
            try:
                pdf.set_font("NotoSans", "B", 12) # Bold, slightly larger font
            except:
                pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, header) # Increased line height for header
            pdf.ln(2) # Small line break after the header

            # Revert to regular font for the content
            try:
                pdf.set_font("NotoSans", "", 11)
            except:
                pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, content)
            pdf.ln(4) # Line break after content for next aspect

        pdf.ln(8) # Ensure space before next manual chunk's analysis

        try:
            pdf.set_font("NotoSans", "I", 9)
        except:
            pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "-" * 100, align='C')
        pdf.ln(8)

    pdf.output(output_filepath)

# --- Helper Function to write individual report to a PDF (MODIFIED) ---
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
        pdf.set_font("NotoSans", "BI", 18) 
    except:
        pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Compliance Analysis for: {original_filename}", align='C') 
    pdf.ln(10)

    # Split the LLM response by aspect headers to apply formatting
    aspect_sections = re.split(r'(\n\*\*Compliance Aspect:.*? \*\*)\n', llm_response_content, flags=re.DOTALL)
    
    if aspect_sections and aspect_sections[0].strip():
        try:
            pdf.set_font("NotoSans", "", 10)
        except:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, aspect_sections[0].strip())

    for i in range(1, len(aspect_sections), 2):
        header = aspect_sections[i].strip()
        content = aspect_sections[i+1].strip() if i+1 < len(aspect_sections) else ""

        # Apply bold and slightly larger font for the aspect header
        pdf.ln(4)
        try:
            pdf.set_font("NotoSans", "B", 11) # Slightly smaller for individual reports
        except:
            pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, header)
        pdf.ln(2)

        # Revert to regular font for the content
        try:
            pdf.set_font("NotoSans", "", 10)
        except:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, content)
        pdf.ln(4)

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
    
    # --- 1. Load and Index Guidelines Document ---
    guidelines_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, GUIDELINES_DOC_NAME)
    if not os.path.exists(guidelines_filepath):
        print(f"Error: Guidelines document '{GUIDELINES_DOC_NAME}' not found at '{DOCUMENT_LIBRARY_PATH}'. Please ensure it exists. Exiting.")
        exit()
    
    raw_guidelines_docs, _ = load_document_content(guidelines_filepath)
    if not raw_guidelines_docs:
        print(f"Error: No content loaded from guidelines document '{GUIDELINES_DOC_NAME}'. Exiting.")
        exit()

    # Chunk the guidelines for the vector store
    guidelines_chunks = split_documents_into_chunks(raw_guidelines_docs, chunk_size=700, chunk_overlap=150) 
    guidelines_vectorstore = create_vector_store(guidelines_chunks, embeddings)
    print(f"Guidelines document '{GUIDELINES_DOC_NAME}' processed and indexed into vector store.")


    compiled_reports_list = [] 

    if not LIST_OF_MANUAL_FILES:
        print("Error: LIST_OF_MANUAL_FILES is empty. Please add file paths to process. Exiting.")
        exit()

    output_directory_base = "./OM_compliance_reports_OVERNIGHT_VERS" 
    os.makedirs(output_directory_base, exist_ok=True)

    individual_reports_dir = os.path.join(output_directory_base, "individual_compliance_reports")
    os.makedirs(individual_reports_dir, exist_ok=True)
    print(f"Individual compliance reports will be saved in: {individual_reports_dir}")


    for chunk_filename_short in LIST_OF_MANUAL_FILES:
        full_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, chunk_filename_short)
        
        current_manual_chunk_docs, loaded_filename = load_document_content(full_filepath)

        if not current_manual_chunk_docs: 
            print(f"Skipping processing for {chunk_filename_short} due to loading error or empty content.")
            continue 

        manual_chunk_full_text = "\n\n".join([doc.page_content for doc in current_manual_chunk_docs])

        print(f"Manual Chunk {loaded_filename} loaded with {len(manual_chunk_full_text)} characters.")

        print(f"\n--- Starting Compliance Analysis for {loaded_filename} ---")

        # Perform compliance check, passing TOP_K_GUIDELINES
        compliance_result_text = check_manual_compliance(
            manual_chunk_full_text, 
            loaded_filename, 
            guidelines_vectorstore, 
            llm, 
            COMPLIANCE_ASPECTS_TO_CHECK,
            TOP_K_GUIDELINES 
        )

        compiled_reports_list.append({"chunk_filename": loaded_filename, "llm_response": compliance_result_text})
        
        base_name = os.path.splitext(loaded_filename)[0]
        individual_pdf_output_filename = os.path.join(individual_reports_dir, f"compliance_report_for_{base_name}.pdf")
        
        write_individual_response_to_pdf(
            compliance_result_text, 
            loaded_filename, 
            individual_pdf_output_filename
        )
        print(f"Saved individual compliance report for '{loaded_filename}' to: {individual_pdf_output_filename}")

        print(f"\n{'='*20} End of Analysis for '{loaded_filename}' {'='*20}\n")
    
    print("\n--- Full Compiled Compliance Report (Console Preview) ---")
    for chunk_report in compiled_reports_list:
        print(f"\n### Compliance Report for Chunk: {chunk_report['chunk_filename']}\n")
        print(chunk_report["llm_response"])
        print("\n--------------------------------------------------")

    consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    try:
        write_report_to_pdf(compiled_reports_list, consolidated_pdf_filename)
        print(f"Successfully saved consolidated compliance report to PDF: {consolidated_pdf_filename}")
    except Exception as e:
        print(f"Error saving consolidated PDF report: {e}")

    print("\nProgram Finished.")