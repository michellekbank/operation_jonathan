# Designed by: Michelle Bank and Carter Musheno
#     for assistance, please contact: mbank@andrew.cmu.edu

#     By the MOU signed by the appropriate parties including the student consultant and representatives from ROPSSA/HCF and Carnegie Mellon, this 
#     program is intellectual property of Michelle Bank and Carter Musheno. ROPSSA and HCF have a worldwide, non-exclusive, royalty-free right and license to copy,
#     modify, publish, distribute, and otherwise use the program and its documentation for purposes consistent with ROPSSA and HCF's mission and 
#     status as an agency.

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

print("Starting ROPSSA Operations Manual Compliance Checker...") # Debugging message to indicate the script has started

# --- Configuration Parameters ---
# These parameters can be adjusted based on the specific requirements of the compliance check
TOP_K_GUIDELINES = 7 # This is the number of top relevant guideline chunks to retrieve for each compliance aspect check
TEMPERATURE = 0.1 # This controls the randomness of the LLM's responses. Lower values make it more deterministic and less creative.
# if you want more creative responses, you can increase this value (e.g., 0.5 or 0.7), but for compliance checks, a lower value is usually better.

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./operations_manual_chunks/Operational Rules and Procedures"

GUIDELINES_DOC_NAME = "41 PNCA 2025.pdf" # Name of your guidelines document

# -- list of directory paths to the individual files --

# This is a list of the manual files to be processed. Each file should be in the DOCUMENT_LIBRARY_PATH directory.
    # additionally, the files should be in the format of .docx, .pdf, or .txt.
    # the files should also be around 5000 characters or less in length, as the LLM can handle up to 8000 characters per prompt.
    # Longer files may result in suboptimal performance or errors.
LIST_OF_MANUAL_FILES = ["Section 101-112.docx",
                        "section 201-202.docx", "section 203–204.docx",
                        "section 205–206.5.docx", "section 304.docx",
                        "section 323-325.docx", "section 506-510.docx",
                        "section 707-711.docx", "sections 206.5A–206.5B.docx",
                        "sections 207–213.docx", "sections 214–215.docx",
                        "sections 216–218.docx", "sections 219–220.docx",
                        "sections 301–303.docx", 
                        "sections 305–309.docx",
                        "sections 310–317.docx", "sections 318–322.docx",
                        "sections 326–330.docx", 
                        "sections 401–407.docx",
                        "sections 501–505.docx", "sections 601–603.docx",
                        "sections 701–706.docx", "sections 801–807.docx",
                        "sections 901–907.docx"
                       ]


# Define specific aspects for targeted compliance checks. This is a list of compliance aspects that the LLM will consider to check against the guidelines.
#     NOTE: NOT ALL OF THESE ASPECTS WILL BE CHECKED FOR EACH MANUAL CHUNK!
#     These aspects are derived from the guidelines document and represent key areas of compliance that need to be verified.
#     The LLM will analyze each manual chunk against the MOST RELEVANT of these aspects and provide a compliance report.
COMPLIANCE_ASPECTS_TO_CHECK = [
        "Establishment and Legal Basis of the Social Security and Healthcare Financing System",
        "Functions, members, and procedures of the Social Security Board",
        "Actuarial Soundness and Sustainability",
        "Fund Reserves and Solvency Requirements",
        "Audit Requirements and External Oversight",
        "Duties, functions, appointment, legal foundations of the Social Security Administrator",
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

# This function uses the LLM to identify which compliance aspects are relevant to a given manual chunk.
#     if there is a specific aspect you have changed in the manual chunk, you can include it manually after the LLM has selected the relevant aspects.
def get_llm_relevant_aspects(manual_chunk_content, available_aspects, llm_model):
    aspects_list_str = "\n".join([f"- {a}" for a in available_aspects])

    prompt = f"""
    You are an expert in regulatory compliance and document analysis.
    Your task is to identify which of the provided 'Predefined Compliance Aspects' are directly and substantively addressed, discussed, or highly relevant within the 'Operations Manual Chunk'. 

    **Instructions:**
    1.  Review the 'Operations Manual Chunk' carefully.
    2.  For each aspect in the 'Predefined Compliance Aspects' list, determine if it is explicitly covered or significantly discussed in the chunk.
    3.  **Strictly adhere:** If an aspect is only mentioned in passing, is vaguely related, or is not the primary focus, DO NOT include it. Prioritize aspects that are clearly the subject of content within the manual chunk. Keep your list short, around 5-10 items. Items in the list can encapsulate more conceptually, but there should only be a few items in the list.
    4.  Return your answer as a list of the exact aspect names **SEPARATED BY NEWLINES** from the 'Predefined Compliance Aspects' list.
    5.  If absolutely no aspects from the list are relevant, respond with **"NONE"**.

    Additionally, if there are policies or procedures not included in the manual chunk that you believe should be included, make note of these. This will help ensure that the compliance check is thorough and comprehensive.

    ---
    **Operations Manual Chunk:**
    {manual_chunk_content}
    ---

    **Predefined Compliance Aspects:**
    {aspects_list_str}
    ---

    **Relevant Aspects (each on a new line):**
    """

    try:
        print("  LLM: Identifying relevant aspects for chunk...")
        start_time = time.time()
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        print(f"  LLM aspect selection took {elapsed:.2f} seconds.")
        selected_aspects_raw = response.content.strip()

        if selected_aspects_raw.upper() == "NONE" or not selected_aspects_raw:
            return []

        # Parse the newline-separated list
        selected_aspects = [aspect.strip() for aspect in selected_aspects_raw.split('\n')]

        # IF YOU WOULD LIKE A SPECIFIC ASPECT TO BE INCLUDED, you can append it here in selected_aspects.

        return selected_aspects
    except Exception as e:
        print(f"Error during LLM aspect selection for chunk: {e}")
        return [] # Return empty list on error

# Checks the compliance of a single manual chunk against guidelines from a vector store, focusing on the specified compliance aspects.
#     This is the main function that performs the compliance check for each manual chunk.
#     It retrieves relevant guidelines for each aspect, constructs a detailed prompt for the LLM, and processes the response.
#     If you want to modify the prompt or the compliance check logic, you can do so here.
def check_manual_compliance(manual_chunk_full_text, manual_chunk_filename, guidelines_vectorstore, llm_model, compliance_aspects_for_this_chunk, top_k_value):
    print(f"\n--- Checking Compliance for Manual Chunk: '{manual_chunk_filename}' ---")
    compliance_report_content = [] # Store parts of the report

    # Iterating through each compliance aspect for a targeted check
    if not compliance_aspects_for_this_chunk:
        compliance_report_content.append("No specific compliance aspects were identified as relevant to this manual chunk by the LLM. Cannot perform targeted checks.")
        print("No specific compliance aspects found as relevant by LLM.")
        return "\n".join(compliance_report_content)

    for aspect in compliance_aspects_for_this_chunk:
        print(f"  Checking aspect: '{aspect}'...")

        # Retrieve relevant sections from the guidelines document based on the aspect query
        # Use the helper function to get relevant guidelines
        print(f"   Retrieving relevant guidelines for aspect: '{aspect}'...")
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
            start_time = time.time()
            print(f"    Prompting LLM for compliance analysis on aspect: '{aspect}'...")
            response = llm_model.invoke(prompt)
            elapsed = time.time() - start_time
            print(f"    Compliance LLM call took {elapsed:.2f} seconds.")
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n{response.content.strip()}")
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
        pdf.add_font("NotoSans", "", regular_font_path, uni=True)
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
    pdf.multi_cell(0, 12, "ROPSSA Operations Manuals - Consolidated Compliance Report", align='C')
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

        aspect_sections = re.split(r'(\n\*\*Compliance Aspect:.*? \*\*)\n', llm_response_content, flags=re.DOTALL)

        if aspect_sections and aspect_sections[0].strip():
            try:
                pdf.set_font("NotoSans", "", 11)
            except:
                pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, aspect_sections[0].strip())

        for i in range(1, len(aspect_sections), 2):
            header = aspect_sections[i].strip()
            content = aspect_sections[i+1].strip() if i+1 < len(aspect_sections) else ""

            pdf.ln(4)
            try:
                pdf.set_font("NotoSans", "B", 12)
            except:
                pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, header)
            pdf.ln(2)

            try:
                pdf.set_font("NotoSans", "", 11)
            except:
                pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, content)
            pdf.ln(4)

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
        pdf.set_font("NotoSans", "BI", 18)
    except:
        pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Compliance Analysis for: {original_filename}", align='C')
    pdf.ln(10)

    # --- MODIFIED: Apply formatting to aspect headers in individual reports ---
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

        pdf.ln(4)
        try:
            pdf.set_font("NotoSans", "B", 11)
        except:
            pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, header)
        pdf.ln(2)

        try:
            pdf.set_font("NotoSans", "", 10)
        except:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, content)
        pdf.ln(4)
    # --- END MODIFIED ---

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

    # --- 2. Chunk the guidelines for the vector store ---
    guidelines_chunks = split_documents_into_chunks(raw_guidelines_docs, chunk_size=700, chunk_overlap=150)
    print("Creating Vector Store For Guidelines Document...")
    guidelines_vectorstore = create_vector_store(guidelines_chunks, embeddings)
    print(f"Guidelines document '{GUIDELINES_DOC_NAME}' processed and indexed into vector store.")

    # --- 3. Setup to call the compliance checker ---
    compiled_reports_list = [] # For the consolidated report

    if not LIST_OF_MANUAL_FILES:
        print("Error: LIST_OF_MANUAL_FILES is empty. Please add file paths to process. Exiting.")
        exit()

    output_directory_base = "./Manual_Lightweight_Tool_Reports" # Base directory for output reports
    os.makedirs(output_directory_base, exist_ok=True)

    individual_reports_dir = os.path.join(output_directory_base, "individual_compliance_reports")
    os.makedirs(individual_reports_dir, exist_ok=True)
    print(f"Individual compliance reports will be saved in: {individual_reports_dir}")

    # --- 4. Process each manual chunk file and update reports ---
    for chunk_filename_short in LIST_OF_MANUAL_FILES:
        full_filepath = os.path.join(DOCUMENT_LIBRARY_PATH, chunk_filename_short)

        # Load the content of the current manual chunk
        current_manual_chunk_docs, loaded_filename = load_document_content(full_filepath)

        if not current_manual_chunk_docs:
            print(f"Skipping processing for {chunk_filename_short} due to loading error or empty content.")
            continue

        # Join the page_content of all documents in the chunk into a single string for the LLM
        manual_chunk_full_text = "\n\n".join([doc.page_content for doc in current_manual_chunk_docs])

        print(f"Manual Chunk {loaded_filename} loaded with {len(manual_chunk_full_text)} characters.")

        # --- NEW: LLM-based aspect selection ---
        print(f"\n--- LLM-based Aspect Selection for {loaded_filename} ---")
        relevant_aspects_for_this_chunk = get_llm_relevant_aspects(
            manual_chunk_full_text,
            COMPLIANCE_ASPECTS_TO_CHECK,
            llm
        )

        if not relevant_aspects_for_this_chunk:
            print(f"LLM found no relevant compliance aspects for '{loaded_filename}'. Skipping compliance check for this chunk.")
            compliance_result_text = f"No relevant compliance aspects found for '{loaded_filename}' by LLM filtering."
            compiled_reports_list.append({"chunk_filename": loaded_filename, "llm_response": compliance_result_text})
            # Save an empty/placeholder individual report for completeness
            base_name = os.path.splitext(loaded_filename)[0]
            individual_pdf_output_filename = os.path.join(individual_reports_dir, f"compliance_report_for_{base_name}.pdf")
            write_individual_response_to_pdf(
                compliance_result_text,
                loaded_filename,
                individual_pdf_output_filename
            )
            continue


        print(f"LLM identified relevant aspects for '{loaded_filename}':")
        for aspect in relevant_aspects_for_this_chunk:
            print(f" - {aspect}")

        print(f"\n--- Starting Compliance Analysis for {loaded_filename} ---")

        # Perform compliance check ONLY with the LLM-identified relevant aspects
        compliance_result_text = check_manual_compliance(
            manual_chunk_full_text,
            loaded_filename,
            guidelines_vectorstore,
            llm,
            relevant_aspects_for_this_chunk, # Pass the LLM-filtered list
            TOP_K_GUIDELINES
        )

        # Store the result for the consolidated report
        compiled_reports_list.append({"chunk_filename": loaded_filename, "llm_response": compliance_result_text})

        # --- Save each individual LLM response (compliance report) to its own PDF ---
        base_name = os.path.splitext(loaded_filename)[0]
        individual_pdf_output_filename = os.path.join(individual_reports_dir, f"compliance_report_for_{base_name}.pdf")

        write_individual_response_to_pdf(
            compliance_result_text,
            loaded_filename,
            individual_pdf_output_filename
        )
        print(f"Saved individual compliance report for '{loaded_filename}' to: {individual_pdf_output_filename}")

        print(f"\n{'='*20} End of Analysis for '{loaded_filename}' {'='*20}\n")

    # --- 5. Compile and Save the Consolidated Compliance Report ---
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