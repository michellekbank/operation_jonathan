import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from fpdf import FPDF
from datetime import datetime

# --- SETUP ---

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "mistral:7b-instruct-q4_k_m"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
TOP_K_GUIDELINES = 5
TEMPERATURE = 0.1

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks/Operational Rules and Procedures"

GUIDELINES_DOC_NAME = "41 PNCA 2025.pdf" # Name of your guidelines document
# -- organization names --
ORG_A_NAMES = ["ROPSSA", "Republic of Palau Social Security Administration", "SSA", 
               "Social Security Administration", "Social Security"]
ORG_B_NAMES = ["HCF", "Health Care Fund", "Republic of Palau Health Care Fund", 
               "Healthcare Fund", "ROPHCF"]

# -- list of directory paths to the individual files --
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
                         "sections 901–907.docx"]


# Define specific aspects/questions for targeted compliance checks
# These are broader themes that the LLM will focus on when comparing to guidelines
# Replace this with actual aspects you want to check!
COMPLIANCE_ASPECTS_TO_CHECK = ["Establishment and Legal Basis of the Social Security Board and Healthcare Financing System",
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
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=TEMPERATURE)

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# -- FUNCTIONS --

# Loads the entire text content from a single document.
# Returns a list of LangChain Document objects (one per page) and the filename.
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

# Splits a list of LangChain Document objects into smaller chunks.
# This is used for the guidelines document to create a vector store.
def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# Creates a FAISS vector store from document chunks and embeddings
def create_vector_store(chunks, embeddings_model):
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

# Helper function to retrieve and format relevant guideline chunks.
# Returns a formatted string of guidelines and a boolean indicating if any were found.
def get_relevant_guideline_context(query_aspect, guidelines_vectorstore, top_k_value):
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

# Checks the compliance of a single manual chunk against guidelines from a vector store,
# focusing on the specified compliance aspects.
def check_manual_compliance(manual_chunk_full_text, manual_chunk_filename, guidelines_vectorstore, llm_model, compliance_aspects, top_k_value):
    print(f"\n--- Checking Compliance for Manual Chunk: '{manual_chunk_filename}' ---")
    compliance_report_content = [] # Store parts of the report

    # Combine organization names for context in the prompt
    org_a_all_names = ", ".join(ORG_A_NAMES)
    org_b_all_names = ", ".join(ORG_B_NAMES)

    # Iterating through each compliance aspect for a targeted check
    if not compliance_aspects:
        compliance_report_content.append("No specific compliance aspects defined. Cannot perform targeted checks.")
        print("No specific compliance aspects defined.")
        return "\n".join(compliance_report_content)

    for aspect in compliance_aspects:
        print(f"  Checking aspect: '{aspect}'...")
        
        # Retrieve relevant sections from the guidelines document based on the aspect query
       # Use the helper function to get relevant guidelines
        guidelines_context, guidelines_found = get_relevant_guideline_context(aspect, guidelines_vectorstore, top_k_value)
        
        if not guidelines_found:
            compliance_report_content.append(
                f"\n**Compliance Aspect: {aspect}**\n"
                f"  - Status: NOT APPLICABLE (No relevant guidelines found for this aspect)."
            )
            continue

        # --- LLM Prompt for Compliance Check ---
        prompt = f"""
        You are a highly analytical compliance officer. Your task is to meticulously evaluate whether the
        'Operations Manual Chunk' explicitly complies with the 'Relevant Guidelines' provided,
        specifically focusing on the aspect: "{aspect}".

        **CRITICAL INSTRUCTIONS FOR ANALYSIS AND REPORTING:**
        1.  **Compliance Status:** Begin your response for this aspect with one of the following:
            * **COMPLIANT:** If the manual chunk fully meets the guideline.
            * **NON-COMPLIANT:** If the manual chunk clearly violates or contradicts the guideline.
            * **PARTIALLY COMPLIANT:** If the manual chunk addresses the guideline but has gaps, ambiguities, or partial adherence.
            * **NOT ADDRESSED:** If the manual chunk does not mention or cover this guideline aspect at all.
        2.  **Explanation & Reasoning:** Provide a concise, objective explanation for your status determination.
        3.  **Verbatim Citations (Crucial):** You MUST cite specific, verbatim phrases or sentences from **both** the 'Operations Manual Chunk' and the 'Relevant Guidelines' to support your reasoning. For each citation, include its original source (e.g., "Manual: '...' (from Section X.Y)", "Guideline: '...' (from Page Z)"). These citations are paramount for traceability.

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

# --- Helper Functions to write the report to PDFs ---
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
    pdf.multi_cell(0, 12, "ROPSSA Operations Manuals - Compliance Report", align='C') # Changed title
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
        pdf.multi_cell(0, 10, f"Compliance Analysis for Manual Chunk: {chunk_filename}", 0, 'L') # Changed title
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
    pdf.multi_cell(0, 10, f"Compliance Analysis for: {original_filename}", align='C') # Changed title
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
    guidelines_chunks = split_documents_into_chunks(raw_guidelines_docs, chunk_size=700, chunk_overlap=150) # Adjust chunking for guidelines
    guidelines_vectorstore = create_vector_store(guidelines_chunks, embeddings)
    print(f"Guidelines document '{GUIDELINES_DOC_NAME}' processed and indexed into vector store.")


    compiled_reports_list = [] # For the consolidated report

    if not LIST_OF_MANUAL_FILES:
        print("Error: LIST_OF_MANUAL_FILES is empty. Please add file paths to process. Exiting.")
        exit()

    output_directory_base = "./OM_compliance_reports" # New output directory for compliance
    os.makedirs(output_directory_base, exist_ok=True)

    individual_reports_dir = os.path.join(output_directory_base, "individual_compliance_reports")
    os.makedirs(individual_reports_dir, exist_ok=True)
    print(f"Individual compliance reports will be saved in: {individual_reports_dir}")


    for chunk_filename_short in LIST_OF_MANUAL_FILES:
        full_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, chunk_filename_short)
        
        # Load the content of the current manual chunk
        # load_document_content returns a list of Document objects (one per page of the chunk file)
        current_manual_chunk_docs, loaded_filename = load_document_content(full_filepath)

        if not current_manual_chunk_docs: # Check if list is empty
            print(f"Skipping processing for {chunk_filename_short} due to loading error or empty content.")
            continue 

        # Join the page_content of all documents in the chunk into a single string for the LLM
        manual_chunk_full_text = "\n\n".join([doc.page_content for doc in current_manual_chunk_docs])

        print(f"Manual Chunk {loaded_filename} loaded with {len(manual_chunk_full_text)} characters.")

        print(f"\n--- Starting Compliance Analysis for {loaded_filename} ---")

        # Perform compliance check
        compliance_result_text = check_manual_compliance(
            manual_chunk_full_text, 
            loaded_filename, 
            guidelines_vectorstore, 
            llm, 
            COMPLIANCE_ASPECTS_TO_CHECK,
            TOP_K_GUIDELINES
        )

        # Store the result for the consolidated report
        compiled_reports_list.append({"chunk_filename": loaded_filename, "llm_response": compliance_result_text})
        
        # --- Save each individual LLM response (compliance report) to its own PDF ---
        base_name = os.path.splitext(loaded_filename)[0]
        individual_pdf_output_filename = os.path.join(individual_reports_dir, f"compliance_report_for_{base_name}.pdf")
        
        write_individual_response_to_pdf(
            compliance_result_text, 
            loaded_filename, # Pass the original filename for the PDF title
            individual_pdf_output_filename
        )
        print(f"Saved individual compliance report for '{loaded_filename}' to: {individual_pdf_output_filename}")

        print(f"\n{'='*20} End of Analysis for '{loaded_filename}' {'='*20}\n")
    
    print("\n--- Full Compiled Compliance Report (Console Preview) ---")
    for chunk_report in compiled_reports_list:
        print(f"\n### Compliance Report for Chunk: {chunk_report['chunk_filename']}\n")
        print(chunk_report["llm_response"])
        print("\n--------------------------------------------------")

    # Save the consolidated compliance report
    consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    try:
        write_report_to_pdf(compiled_reports_list, consolidated_pdf_filename)
        print(f"Successfully saved consolidated compliance report to PDF: {consolidated_pdf_filename}")
    except Exception as e:
        print(f"Error saving consolidated PDF report: {e}")

    print("\nProgram Finished.")