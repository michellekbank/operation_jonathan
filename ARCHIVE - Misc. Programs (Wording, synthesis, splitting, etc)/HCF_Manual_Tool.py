# Designed by: Michelle Bank and Carter Musheno
# Programmed and developed by: Michelle Bank
#     for assistance, please contact: mbank@andrew.cmu.edu

#     By the MOU signed by the appropriate parties including the student consultant and representatives from ROPSSA/HCF and Carnegie Mellon, this 
#     program is intellectual property of Michelle Bank and Carter Musheno. ROPSSA and HCF have a worldwide, non-exclusive, royalty-free right and license to copy,
#     modify, publish, distribute, and otherwise use the program and its documentation for purposes consistent with ROPSSA and HCF's mission and 
#     status as an agency.

#     This program is designed to be used by ROPSSA & HCF for the purpose of checking compliance of the Operations Manual and other various documents
#     against a set of guidelines.

# Description: This script is designed to check a small section of a document against a set of guidelines.
#     It analyzes the section to determine appropriate compliance aspects then loads relevant chunks from the guidelines document and checks the section
#     against the appropriate guidelines.

# Intented Use: It is designed to be used when a small chunk manual is updated and needs to be checked against a set of guidelines.
#     Runtime for this script is much shorter than the full compliance checker script, as it only checks the most relevant aspects of the guidelines.
#     This makes it an effective tool for quickly checking compliance of small sections of the manual without needing to run the full compliance checker.
#     Consider specifying what was specifically updated in the manual chunk to help the LLM focus on the most relevant aspects. This can be done by including this
#     information in the list of "aspects" to check, or by providing a brief description of the changes made to the manual chunk in the LLM prompt.

# Troubleshooting: If you encounter issues with the script, please check the following:
#   1. Ensure that the LM Studio is running and accessible at the specified base URL.
#   2. Ensure that the guidelines document and manual files are present in the specified directories.
#   3. Ensure that the required Python packages are installed and up-to-date.
#   4. If you encounter any errors, please check the console output for debugging messages and error messages.

# --- import necessary libraries ---

import os
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from fpdf import FPDF
from datetime import datetime
import time

print("Starting HCF Operations Manual Synthesis...") # Debugging message to indicate the script has started

# --- Configuration Parameters ---
# These parameters can be adjusted based on the specific requirements of the compliance check
TOP_K_GUIDELINES = 7 # This is the number of top relevant guideline chunks to retrieve for each compliance aspect check
TEMPERATURE = 0.1 # This controls the randomness of the LLM's responses. Lower values make it more deterministic and less creative.
# if you want more creative responses, you can increase this value (e.g., 0.5 or 0.7), but for compliance checks, a lower value is usually better.

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks/Operational Rules and Procedures"

GUIDELINES_DOC_NAME = "41 PNCA 2025.pdf" # Name of your guidelines document

# -- directory paths to the template --

# This is a template for the LLM to follow. Tje file should be in the DOCUMENT_LIBRARY_PATH directory.
    # additionally, the file should be in the format of .docx, .pdf, or .txt.
    # the file should also be around 5000 characters or less in length, as the LLM can handle up to 8000 characters per prompt.

TEMPLATE_FILE = "section 304.docx"

# This will populate as the program runs!
LIST_OF_MANUAL_FILES = []


# Define specific aspects for targeted compliance checks. This is a list of compliance aspects that the LLM will consider to check against the guidelines.
#     NOTE: NOT ALL OF THESE ASPECTS WILL BE CHECKED FOR EACH MANUAL CHUNK!
#     These aspects are derived from the guidelines document and represent key areas of compliance that need to be verified.
#     The LLM will analyze each manual chunk against the MOST RELEVANT of these aspects and provide a compliance report.
COMPLIANCE_ASPECTS_TO_CHECK = [
        "Estalishment and Legal Basis of the Healthcare Financing System",
        "Functions, members, and procedures of the National Healthcare Financing Governing Committee or the “Committee”",
        "Actuarial Soundness and Sustainability",
        "Fund Reserves and Solvency Requirements",
        "Audit Requirements and External Oversight (ensuring proper management of the healthcare fund)",
        "Duties, functions, appointment, legal foundations of the Social Security Administrator as they pertain to overseeing the healthcare financing system",
        "Financial reporting and budget (Direct financial operation of the healthcare system)",
        "Specific funding sources and collection mechanisms for the healthcare financing system",
        "healthcare provider payment and contracting mechanisms",
        "Quality assurance and patient safety standards"
        "Governance Structure and Oversight Mechanisms (including the National Healthcare Financing Governing Committee or the “Committee”)",
        "Enrollment and eligibility criteria (healthcare system)",
        "Data management, security, and information sharing mechanisms and policies (health specific)",
        "Appeals and Dispute Resolution Mechanisms",
        "Beneficiary Rights and Responsibilities",
        "Investment Policies, Portfolio Management, and Performance Reporting ( For managing the assets of the healthcare fund.)",
        "Incomes and contributions or payments ( how the healthcare system is financed)",
        "Claims",
        "Aspects of health insurance, including benefits, exclusions, reimbursements, and subscriptions",
        "Privacy (especially for health information)",
        "Employee offenses and penalties including fraud, failure to report or pay, false claims ()",
        "Enforcement Powers and Sanctions for Non-Compliance (beyond just offenses) within the healthcare system",
        "Succession and transfer of medical savings account after death (health specific)",
        "The keeping of accounts and reports (especially for the management of the healthcare fund)"
]

# --- INITIALIZATION OF LLM ---

# Ensure the model is running in LM studio on your computer
lm_studio_base_url = "http://localhost:1234/v1"  # Default LM Studio URL. Depending on your setup, you may need to change this. Check LM Studio settings.
print(f"Initializing LLM with LM Studio (local OpenAI-compatible API) from: {lm_studio_base_url}")
llm = ChatOpenAI(base_url=lm_studio_base_url, api_key="lm-studio", model="local-model", temperature=TEMPERATURE)

# Note: Embeddings are still used for the guidelines vector store.
print(f"Initializing Embeddings with CPU-Based HuggingFaceEmbeddings.")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# --- FUNCTIONS ---

# Loads the entire text content from a single document and returns a list of LangChain Document objects (one per page) and the filename.
#   the accepted file types are .pdf, .docx, and .txt. If you would like to add more file types, you can add them to the if-elif statements below.
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
#    This is used for the guidelines document to create a vector store.
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

# Retrieves and formats relevant guideline chunks.
#     Returns a formatted string of guidelines and a boolean indicating if any were found.
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

# --- NEW FUNCTION: Synthesizes a new manual chunk based on guidelines for a specific aspect ---
def synthesize_manual_chunk(aspect, guidelines_vectorstore, llm_model, template_content, top_k_value):
    print(f"\n--- Synthesizing Manual Section for Aspect: '{aspect}' ---")

    # Retrieve relevant sections from the guidelines document based on the aspect query
    print(f"   Retrieving relevant guidelines for aspect: '{aspect}'...")
    guidelines_context, guidelines_found = get_relevant_guideline_context(aspect, guidelines_vectorstore, top_k_value)

    if not guidelines_found:
        print(f"   No relevant guidelines found for aspect: '{aspect}'. Skipping synthesis.")
        return f"\n## {aspect}\n\n[SECTION PENDING: No sufficient guidelines found for this aspect.]\n"

    # --- LLM Prompt for Manual Synthesis ---
    prompt = f"""
    You are an expert policy writer and content creator for the Republic of Palau Social Security Administration (ROPSSA) and its Healthcare Fund (HCF). Your task is to draft a clear, comprehensive, and operational section of the **Healthcare Fund Operations Manual** that precisely details the policies and procedures related to the aspect: "{aspect}".

    **Instructions:**
    1.  **Draft a complete and clear manual section** for the given aspect.
    2.  **Strictly adhere to the provided 'Relevant Guidelines'**. Every statement in your manual section must be directly supported by or derived from the guidelines. Do not introduce information not present in the guidelines.
    3.  **Maintain an authoritative, operational, and unambiguous tone** suitable for a formal operations manual.
    4.  **Follow the template structure** and used in the provided manual section template file. Ensure your section is well-organized, with clear headings and subheadings as needed.
    5.  **Do not include any compliance analysis or explanations. Just provide the manual text.
    6.  **If the provided guidelines for the aspect are insufficient or vague** to create a detailed manual section, state: "Insufficient guidelines provided for a detailed manual section on: {aspect}."
    7.  **Provide citations for any specific guidelines used** in the manual section, referencing the guideline excerpt number or source document.

    **Section Template:**
    ---
    {template_content}
    ---

    **Relevant Guidelines (related to '{aspect}'):**
    ---
    {guidelines_context}
    ---

    **Operations Manual Section for '{aspect}':**
    """
    # --- End of LLM Prompt ---

    try:
        start_time = time.time()
        print(f"   Prompting LLM for manual section synthesis on aspect: '{aspect}'...")
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        print(f"   Manual section synthesis took {elapsed:.2f} seconds.")
        # Prepend with a heading for clarity in the final manual
        return f"\n## {aspect}\n\n{response.content.strip()}\n"
    except Exception as e:
        print(f"   Error during manual section synthesis for aspect '{aspect}': {e}")
        return f"\n## {aspect}\n\n[ERROR: Could not synthesize section due to an error: {e}]\n"

# --- Helper Function to write the synthesized manual to PDF ---
def write_synthesized_manual_to_pdf(manual_content, output_filepath, title="ROPSSA Healthcare Fund Operations Manual"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_dir = "./fonts"
    regular_font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    italic_font_path = os.path.join(font_dir, "NotoSans-Italic.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    bolditalic_font_path = os.path.join(font_dir, "NotoSans-BoldItalic.ttf")

    # Add custom fonts if available, otherwise FPDF will use its defaults (like Helvetica)
    if os.path.exists(regular_font_path):
        pdf.add_font("NotoSans", "", regular_font_path, uni=True)
    if os.path.exists(italic_font_path):
        pdf.add_font("NotoSans", "I", italic_font_path, uni=True)
    if os.path.exists(bold_font_path):
        pdf.add_font("NotoSans", "B", bold_font_path, uni=True)
    if os.path.exists(bolditalic_font_path):
        pdf.add_font("NotoSans", "BI", bolditalic_font_path, uni=True)

    # Set title font and write title
    try:
        pdf.set_font("NotoSans", "B", 24)
    except Exception:
        pdf.set_font("Helvetica", "B", 24) # Fallback
    pdf.multi_cell(0, 15, title, align='C')
    pdf.ln(15)

    # Set content font
    try:
        pdf.set_font("NotoSans", "", 12)
    except Exception:
        pdf.set_font("Helvetica", "", 12) # Fallback

    # Process content to handle headings and basic formatting
    lines = manual_content.split('\n')
    for line in lines:
        if line.strip().startswith('## '): # Major headings
            pdf.ln(5) # Add some space before heading
            try:
                pdf.set_font("NotoSans", "B", 16)
            except Exception:
                pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 10, line.strip('# ').strip(), 0, 'L')
            pdf.ln(2) # Add some space after heading
            try:
                pdf.set_font("NotoSans", "", 12) # Reset to normal font for body
            except Exception:
                pdf.set_font("Helvetica", "", 12)
        elif line.strip(): # Regular content
            pdf.multi_cell(0, 7, line.strip(), 0, 'L')
        else: # Empty line for paragraph breaks
            pdf.ln(5)

    try:
        pdf.output(output_filepath)
        print(f"Successfully saved synthesized manual to PDF: {output_filepath}")
    except Exception as e:
        print(f"Error saving synthesized manual to PDF: {e}")


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

    # --- 2. Chunk the guidelines for the vector store ---
    guidelines_chunks = split_documents_into_chunks(raw_guidelines_docs, chunk_size=700, chunk_overlap=150)
    print("Creating Vector Store For Guidelines Document...")
    guidelines_vectorstore = create_vector_store(guidelines_chunks, embeddings)
    print(f"Guidelines document '{GUIDELINES_DOC_NAME}' processed and indexed into vector store.")

     # --- 3. Load the Template File Content ---
    template_full_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, TEMPLATE_FILE)
    if not os.path.exists(template_full_filepath):
        print(f"Error: Template file '{TEMPLATE_FILE}' not found at '{DOCUMENT_LIBRARY_PATH}'. Please ensure it exists. Exiting.")
        exit()
    raw_template_docs, _ = load_document_content(template_full_filepath)
    if not raw_template_docs:
        print(f"Error: No content loaded from template file '{TEMPLATE_FILE}'. Exiting.")
        exit()
    template_content_for_llm = "\n\n".join([doc.page_content for doc in raw_template_docs])
    print(f"Template file '{TEMPLATE_FILE}' loaded with {len(template_content_for_llm)} characters.")

    # --- 4. Synthesize the new Operations Manual ---
    synthesized_manual_sections = []
    print("\n--- Starting Operations Manual Synthesis ---")

    for aspect in COMPLIANCE_ASPECTS_TO_CHECK:
        manual_section_content = synthesize_manual_chunk(
            aspect,
            guidelines_vectorstore,
            llm,  template_content_for_llm,
            TOP_K_GUIDELINES)
        synthesized_manual_sections.append(manual_section_content)
        print(f"\n{'='*20} End of Synthesis for '{aspect}' {'='*20}\n")

    full_synthesized_manual_content = "\n\n".join(synthesized_manual_sections)

    # --- 5. Save the Synthesized Manual to PDF ---
    output_directory = "./Synthesized_Operations_Manuals"
    os.makedirs(output_directory, exist_ok=True)
    synthesized_pdf_filename = os.path.join(output_directory, f"ROPSSA_HCF_Operations_Manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    write_synthesized_manual_to_pdf(
        full_synthesized_manual_content,
        synthesized_pdf_filename,
        title="Healthcare Fund Operations Manual (Generated by AI)" # Clearer title
    )

    print("\nProgram Finished: New Operations Manual Synthesized.")