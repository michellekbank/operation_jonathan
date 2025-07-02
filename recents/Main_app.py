# Designed by: Michelle Bank and Carter Musheno
#     for assistance, please contact: mbank@andrew.cmu.edu

#     By the MOU signed by the appropriate parties including the student consultant and representatives from ROPSSA/HCF and Carnegie Mellon, this 
#     program is intellectual property of Michelle Bank and Carter Musheno. ROPSSA and HCF have a worldwide, non-exclusive, royalty-free right and license to copy,
#     modify, publish, distribute, and otherwise use the program and its documentation for purposes consistent with ROPSSA and HCF's mission and 
#     status as an agency.

# --- import necessary libraries ---
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import os
from datetime import date, datetime
import time
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from fpdf import FPDF
import re
import sys
import traceback

# --------- Configuration Parameters ---------

TOP_K_GUIDELINES = 7 # This is the number of top relevant guideline chunks to retrieve for each compliance aspect check

# --- Temperature ---
# Controls the randomness of the LLM's responses. 
#     Lower values make it more deterministic and less creative.

COMPLIANCE_TEMPERATURE = 0.1 #lower for compliance because we want LLM to be strict and accurate
EDITING_TEMPERATURE = 0.5 # higher for editing because we want LLM to be more creative and flexible

 # Default LM Studio URL. Depending on your setup, you may need to change this. Check LM Studio settings.
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# --------- GLOBAL CONSTANTS ---------

# --- Chunk Sizes ---
# These chunk sizes are used for splitting documents into manageable pieces for processing.
#     They can be adjusted based on the size of the documents you are working with.
WORD_EDITOR_CHUNK_SIZE = 4000 
COMPLIANCE_CHECKER_CHUNK_SIZE = 3000

# --------- GLOBAL VARIABLES FOR GUI STATE ---------
# These variables are used to store the state of the application, such as selected files and LLM/embeddings instances.

# Paths to selected files
selected_guidelines_path = ""
selected_compliance_manual_path = ""
selected_editing_manual_path = ""
selected_section_path = ""
selected_sample_path = ""

# llm stuff
llm = None
embeddings = None

# Program Information logs
compliance_log = None
editor_log = None
section_synthesis_log = None
troubleshooting_log = None

# tab controls
notebook = None
compliance_tab_frame = None
word_editor_tab_frame = None
section_synthesis_tab_frame = None
troubleshooting_tab_frame = None
home_frame = None
tab_frames = {} # A dictionary to hold references to tab frames for easier lookup

# Global references to buttons/labels that need to be enabled/disabled or updated
guidelines_path_label = None
compliance_manual_label = None
compliance_aspects_input = None
start_compliance_button = None
guidelines_button = None
manual_compliance_button = None
start_synthesis_button = None
section_path_label = None
section_button = None
sample_path_label = None
sample_button = None
section_synthesis_button = None
editing_manual_path_label = None
start_editing_button = None
check_button = None

# These functions are used to log messages to the user in the application's log area. It's at the top because it's used by practically every function in this file.
def get_active_tab_log_safe():
    if not notebook or not root.winfo_exists(): # Check if notebook and root exist
        return None

    try:
        current_tab_frame = notebook.nametowidget(notebook.select())

        if current_tab_frame == compliance_tab_frame:
            return compliance_log
        elif current_tab_frame == word_editor_tab_frame:
            return editor_log
        elif current_tab_frame == section_synthesis_tab_frame:
            return section_synthesis_log
        elif current_tab_frame == troubleshooting_tab_frame:
            return troubleshooting_log
    except Exception as e:
        # This catch is mainly for debugging if this function is called unsafely
        print(f"Error getting active tab log (likely from non-main thread or GUI closed): {e}")
    return None

def log(message, target_log):
    def _insert_message():
        log_widget = target_log

        if log_widget is None:
            # If no target specified, try to get the active tab's log.
            # This part MUST also be called via root.after if coming from a non-main thread.
            log_widget = get_active_tab_log_safe()

        if log_widget and log_widget.winfo_exists():
            log_widget.insert(tk.END, message + "\n")
            log_widget.see(tk.END)
        else:
            # Fallback for when no GUI log is available (e.g., app closing, error during init)
            print(f"GUI Log (No target/widget exists): {message}")

    if root and root.winfo_exists():
        # Schedule the _insert_message function to run on the main Tkinter thread
        root.after(0, _insert_message) # 0 delay means as soon as possible
    else:
        # If root window doesn't exist, print to console as a last resort
        print(f"Console Log (GUI not available): {message}")

# These functiosn are used to make sure pathing is correct

def resource_path(relative_path):
#Get absolute path to resource, works for dev and for PyInstaller
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # If not running in a PyInstaller bundle, use the script's directory
        base_path = os.path.abspath(".")
        
    return os.path.join(base_path, relative_path)

#Get path relative to the executable's directory."""
def get_executable_dir_path(relative_path=""):
    if getattr(sys, 'frozen', False): # Check if running as a PyInstaller bundle
        # If frozen (bundled), sys.executable is the path to the exe
        base_path = os.path.dirname(sys.executable)
    else:
        # If not frozen (running from script), use current working directory
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --------- DOCUMENT PROCESSING ---------

# Loads the entire text content from a single document and returns a list of LangChain Document objects (one per page) and the filename.
#   the accepted file types are .pdf, .docx, and .txt. If you would like to add more file types, you can add them to the if-elif statements below.
def load_document_content(filepath, target_log):
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

    log(f"Loading raw content from {os.path.basename(filepath)}...\n", target_log=target_log)
    try:
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"] = os.path.basename(filepath)
        return raw_docs, os.path.basename(filepath)
    except Exception as e:
        log( f"Error loading {os.path.basename(filepath)}: {e}\n", target_log=target_log) 
        return [], None

# Splits a list of LangChain Document objects into smaller chunks.
#    This is used for the guidelines document to create a vector store and when processing large manuals.
def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# Creates a FAISS vector store from document chunks and embeddings
def create_vector_store(chunks, embeddings_model, target_log):
    log("Creating vector store from document chunks...\n", target_log=target_log)
    return FAISS.from_documents(chunks, embeddings_model)

# --------- CORE FUNCTIONS FOR COMPLIANCE CHECKER ---------

# Retrieves and formats relevant guideline chunks.
#     Returns a formatted string of guidelines and a boolean indicating if any were found.
def get_relevant_guideline_context(query_aspect, guidelines_vectorstore, top_k_value):
    relevant_guidelines_docs = guidelines_vectorstore.similarity_search(query_aspect, k=top_k_value)

    if not relevant_guidelines_docs:
        return "", False

    guidelines_context_parts = []
    for i, doc in enumerate(relevant_guidelines_docs):
        page_info = doc.metadata.get('page', 'N/A')
        if doc.metadata.get('source', '').lower().endswith('.pdf') and isinstance(page_info, int):
            page_info = page_info + 1

        guidelines_context_parts.append(
            f"Guideline Excerpt {i+1} (Source: {doc.metadata.get('source', 'Unknown')}, Page: {page_info}):\n"
            f"\"{doc.page_content}\""
        )
    guidelines_context = "\n\n".join(guidelines_context_parts)
    return guidelines_context, True

# This function uses the LLM to identify which compliance aspects are relevant to a given manual chunk.
def get_llm_relevant_aspects(manual_chunk_content, available_aspects, llm_model):
    global compliance_log
    target_log = compliance_log
    aspects_list_str = "\n".join([f"- {a}" for a in available_aspects])

    prompt = f"""
    You are an expert in regulatory compliance and document analysis.
    Your task is to identify which of the provided 'Predefined Compliance Aspects' are directly and substantively addressed, discussed, or highly relevant within the 'Operations Manual Chunk'.
    
    **This list will be used as search queries** to retrieve relevant sections from a comprehensive guidelines document for a detailed compliance check. Therefore, focus on generating aspects that will yield precise and useful guideline content.
        FOR EXAMPLE: 
            *Dates, numbers, and figures are important to include. For example, if the manual states "the maximum fine for employees is $10,000" make sure "Policies on maximum fines for employees" is in the list.
            * Specific inclusions and exclusions are important to include. For example, if the manual states "Self-employed individuals are not subject to this policy" make sure "Policies on self-employed individuals" is in the list.
    **Instructions:**
    1.  Review the portion of the operations manual in "Operations Manual Chunk" carefully.
    2.  Using the aspects in the 'Predefined Compliance Aspects' list as reference, find **explicitly covered or significantly discussed** compliance topics, policies, procedures, or requirements that are **directly relevant** to the manual chunk.
    2.  **Exclude** aspects that are only vaguely related or briefly mentioned. Focus on primary subjects of the chunk.
    3.  **Output Format:** Return your answer as a list of the exact relevant aspect names, **each on a new line**.
    4.  **List Length:** Keep the list short and high-level (aim for 5 items) but allow items to conceptually encapsulate more.
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
        log( "   LLM: Identifying relevant aspects for chunk...\n", target_log)
        start_time = time.time()
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        log(f"   LLM aspect selection took {elapsed:.2f} seconds.\n", target_log)
        selected_aspects_raw = response.content.strip()

        if selected_aspects_raw.upper() == "NONE" or not selected_aspects_raw:
            return []

        selected_aspects = [aspect.strip() for aspect in selected_aspects_raw.split('\n')]
        return selected_aspects
    except Exception as e:
        log(f"Error during LLM aspect selection for chunk: {e}\n", target_log)
        return []

# Checks the compliance of a single manual chunk against guidelines from a vector store, focusing on the specified compliance aspects.
#     This is the main function that performs the compliance check for each manual chunk.
#     It retrieves relevant guidelines for each aspect, constructs a detailed prompt for the LLM, and processes the response.
#     If you want to modify the prompt or the compliance check logic, you can do so here.
def check_manual_compliance(manual_chunk_full_text, manual_chunk_index, guidelines_vectorstore, llm_model, compliance_aspects_for_this_chunk, top_k_value):
    global compliance_log
    target_log = compliance_log
    log(f"\n--- Checking Compliance for Document Chunk: '{manual_chunk_index}' ---\n", target_log)
    compliance_report_content = []
    current_date = date.today().strftime("%B %d, %Y")

    if not compliance_aspects_for_this_chunk:
        compliance_report_content.append("No specific compliance aspects were identified as relevant to this chunk by the LLM. Cannot perform targeted checks.")
        log("No specific compliance aspects found as relevant by LLM.\n", target_log)
        return "\n".join(compliance_report_content)

    for aspect in compliance_aspects_for_this_chunk:
        log(f"   Checking aspect: '{aspect}'...\n", target_log)

        log(f"    Retrieving relevant guidelines for aspect: '{aspect}'...\n", target_log)
        guidelines_context, guidelines_found = get_relevant_guideline_context(aspect, guidelines_vectorstore, top_k_value)

        if not guidelines_found:
            compliance_report_content.append(
                f"\n**Compliance Aspect: {aspect}**\n"
                f"   - Status: NOT APPLICABLE (No relevant guidelines found for this aspect)."
            )
            continue

        prompt = f"""
        You are a highly analytical compliance officer. Your task is to meticulously evaluate whether the
        'Operations Manual Chunk' explicitly complies with the 'Relevant Guidelines' provided,
        specifically focusing on the aspect: "{aspect}".

        **Note the current date: {current_date}.** Consider evolving compliance requirements and that best practices may evolve over time.

        **REPORT INSTRUCTIONS:**
        
        **CRITICAL INSTRUCTIONS FOR ANALYSIS AND REPORTING:**
        1.  **Compliance Status:** Begin your response for this aspect with one of the following:
            * **COMPLIANT:** If the manual chunk fully meets the guideline.
            * **NON-COMPLIANT:** If the manual chunk clearly violates or contradicts the guideline.
            * **PARTIALLY COMPLIANT:** If the manual chunk *attempts to address* the guideline but does so incompletely, vaguely, or with minor deficiencies that prevent full adherence.
            * **NOT ADDRESSED:** If the 'Operations Manual Chunk' **does not contain sufficient information or discussion relevant to this specific guideline aspect**, or if the aspect is entirely absent from the manual's content. Do not use 'PARTIALLY COMPLIANT' if the manual simply lacks content on the topic.

        2.  **Explanation & Reasoning:** Provide a concise, objective explanation for your status determination.

        3.  **Verbatim Citations (ESSENTIAL):** You **must** include direct, verbatim quotes from **both** the 'Operations Manual Chunk' and the 'Relevant Guidelines' to substantiate your reasoning. For each citation, specify its source clearly:
            * **Manual:** "Quote." (from Section/Page X)
            * **Guideline:** "Quote." (from Section/Page Y)
            * **If no direct manual citation for COMPLIANT/NON-COMPLIANT, consider 'NOT ADDRESSED'.**

        ---

        **OPERATIONS MANUAL CHUNK (from {manual_chunk_index}):**
        ---
        {manual_chunk_full_text}
        ---

        **RELEVANT GUIDELINES (for '{aspect}'):**
        ---
        {guidelines_context}
        ---

        **BEGIN ANALYSIS FOR ASPECT: "{aspect}"**
        """

        try:
            start_time = time.time()
            log(f"    Prompting LLM for compliance analysis on aspect: '{aspect}'...\n", target_log)
            response = llm_model.invoke(prompt)
            elapsed = time.time() - start_time
            log(f"    Compliance LLM call took {elapsed:.2f} seconds.\n", target_log)
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n{response.content.strip()}")
        except Exception as e:
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n   - Error during analysis: {e}")
            log(f"   Error analyzing aspect '{aspect}' for '{manual_chunk_index}': {e}\n", target_log)

    return "\n".join(compliance_report_content)

# --------- CORE FUNCTION FOR WORD EDITOR ---------

# Prompts the LLM to edit a chunk of text
#    This is the main function that performs the editing for each manual chunk.
#    If you want to modify the prompt or the editing check logic, you can do so here.
def editor_analyze_single_chunk(chunk_full_text, chunk_index, llm_model): 
    global editor_log
    target_log = editor_log
    log(f"\n--- Analyzing Chunk {chunk_index} ---", target_log)

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
        start_time = time.time()
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        log(f"   LLM editing took {elapsed:.2f} seconds for Chunk {chunk_index}.\n", target_log)
        # Access the content attribute of the AIMessage object
        return {"chunk_id": f"Chunk {chunk_index}", "llm_response": response.content} # MODIFIED LINE
    except Exception as e:
        log(f"Error processing 'Chunk {chunk_index}': {e}\n", target_log)
        return {"chunk_id": f"Chunk {chunk_index}", "llm_response": f"Error processing 'Chunk {chunk_index}': {e}"}

# --------- CORE FUNCTION SECTION SYNTHESIS ---------

# Prompts the LLM to synthesize a section of text based on a sample text and source material.
#    This is the main function that performs the synthesis for each section.
#    Note: this only takes in small documents!!!!!
#    If you want to modify the prompt or the editing check logic, you can do so here.
def section_synthesis(section_text, sample_text, llm_model):
    global section_synthesis_log
    target_log = section_synthesis_log
    synthesis_result = ""
    log(f"\n--- Synthesizing Section... ---\n", target_log)

    if not section_text:
        synthesis_result = "No section text provided for synthesis."
        log("No section text provided for synthesis.\n", target_log)
        return synthesis_result

    if not sample_text:
        synthesis_result = "No sample text provided for style match."
        log("No sample text provided for style match.\n", target_log)
        return synthesis_result

    prompt = f"""
    You are an expert in document synthesis and analysis. Your task is to analyze the provided source text and synthesize it into a concise policy statement in the style of the sample text.

    **Instructions:**
    1. Analyze the section text carefully.
    2. Provide a concise policy statement that captures the key points, themes, and insights from the source text written with the style and organization of the sample text.
    3. DO NOT OMMIT ANY INFORMATION FROM THE SOURCE TEXT! Ensure that your response is clear, coherent, and well-structured.

    **Source Text:**
    {section_text}

    **Sample Text:**
    {sample_text}

    **Your Summary/Analysis:**
    """
    try:
        start_time = time.time()
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        log(f"    LLM section synthesis took {elapsed:.2f} seconds.\n", target_log)
        synthesis_result = response.content.strip()
    except Exception as e:
        synthesis_result = f"Error during section synthesis: {e}"
        log(f"Error during section synthesis: {e}\n", target_log)

    return synthesis_result    

# --------- PDF WRITING FUNCTIONS ---------

# This helper function centralizes PDF object creation and font loading
def setup_pdf_with_fonts(target_log):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_dir = resource_path("fonts")
    
    regular_font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    italic_font_path = os.path.join(font_dir, "NotoSans-Italic.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    bolditalic_font_path = os.path.join(font_dir, "NotoSans-BoldItalic.ttf")

    font_loaded = False
    try:
        if os.path.exists(regular_font_path):
            pdf.add_font("NotoSans", "", regular_font_path, uni=True)
            font_loaded = True
        else:
            log(f"Warning: NotoSans-Regular.ttf not found at {regular_font_path}. Using default font.\n", target_log) 
        if os.path.exists(italic_font_path):
            pdf.add_font("NotoSans", "I", italic_font_path, uni=True)
        if os.path.exists(bold_font_path):
            pdf.add_font("NotoSans", "B", bold_font_path, uni=True)
        if os.path.exists(bolditalic_font_path):
            pdf.add_font("NotoSans", "BI", bolditalic_font_path, uni=True)
    except Exception as e:
        log(f"Error loading custom fonts: {e}. Falling back to default.\n", target_log) 
        font_loaded = False

    return pdf, font_loaded

# --- Used by the editor ---
def write_edited_consolidated_pdf(report_data, original_doc_name, output_filepath):
    global editor_log
    target_log = editor_log
    pdf, font_loaded = setup_pdf_with_fonts(target_log) 
    pdf.add_page()
    try:
        # Use font_loaded flag to determine which font to set
        pdf.set_font("NotoSans", "BI", 20) if font_loaded else pdf.set_font("Helvetica", "B", 20)
    except:
        pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 12, f"CONSOLIDATED REPORT: Edited Content for: {original_doc_name}", align='C')
    pdf.ln(15)

    for chunk_info in report_data:
        chunk_id = chunk_info["chunk_id"]
        display_content = chunk_info["llm_response"]

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16) if font_loaded else pdf.set_font("Helvetica", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Edited Content for {chunk_id}", 0, 'L')
        pdf.ln(4)

        try:
            pdf.set_font("NotoSans", "", 11) if font_loaded else pdf.set_font("Helvetica", "", 11)
        except:
            pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, display_content)
        pdf.ln(8)

        try:
            pdf.set_font("NotoSans", "I", 9) if font_loaded else pdf.set_font("Helvetica", "I", 9)
        except:
            pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "-" * 100, align='C')
        pdf.ln(8)

    pdf.output(output_filepath)

def write_edited_individual_pdf(llm_response_content, chunk_id, original_doc_name, output_filepath):
    global editor_log
    target_log = editor_log
    pdf, font_loaded = setup_pdf_with_fonts(target_log)
    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 18) if font_loaded else pdf.set_font("Helvetica", "B", 18)
    except:
        pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Edited Content for document chunk {chunk_id} from {original_doc_name}", align='C')
    pdf.ln(10)

    # llm_response_content is already a string
    display_content = llm_response_content

    try:
        pdf.set_font("NotoSans", "", 10) if font_loaded else pdf.set_font("Helvetica", "", 10)
    except:
        pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, display_content)
    pdf.ln(5)

    try:
        pdf.output(output_filepath)
    except Exception as e:
        log(f"Error saving individual PDF for {chunk_id} to {output_filepath}: {e}\n", target_log)

# --- Used by the compliance checker ---
def write_report_to_pdf(report_data, output_filepath):
    global compliance_log
    target_log = compliance_log
    pdf, font_loaded = setup_pdf_with_fonts(target_log)
    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 20) if font_loaded else pdf.set_font("Helvetica", "B", 20)
    except: # Fallback if font loading fails even after check
        pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 12, "Consolidated Compliance Report", align='C')
    pdf.ln(15)

    for chunk_info in report_data:
        chunk_filename = chunk_info["chunk_filename"]
        llm_response_content = chunk_info["llm_response"]

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16) if font_loaded else pdf.set_font("Helvetica", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Compliance Analysis for Document Chunk: {chunk_filename}", 0, 'L')
        pdf.ln(4)

        aspect_sections = re.split(r'(\n\*\*Compliance Aspect:.*? \*\*)\n', llm_response_content, flags=re.DOTALL)

        if aspect_sections and aspect_sections[0].strip():
            try:
                pdf.set_font("NotoSans", "", 11) if font_loaded else pdf.set_font("Helvetica", "", 11)
            except:
                pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, aspect_sections[0].strip())

        for i in range(1, len(aspect_sections), 2):
            header = aspect_sections[i].strip()
            content = aspect_sections[i+1].strip() if i+1 < len(aspect_sections) else ""

            pdf.ln(4)
            try:
                pdf.set_font("NotoSans", "B", 12) if font_loaded else pdf.set_font("Helvetica", "B", 12)
            except:
                pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, header)
            pdf.ln(2)

            try:
                pdf.set_font("NotoSans", "", 11) if font_loaded else pdf.set_font("Helvetica", "", 11)
            except:
                pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, content)
            pdf.ln(4)

        pdf.ln(8)

        try:
            pdf.set_font("NotoSans", "I", 9) if font_loaded else pdf.set_font("Helvetica", "I", 9)
        except:
            pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "-" * 100, align='C')
        pdf.ln(8)

    pdf.output(output_filepath)

def write_individual_response_to_pdf(llm_response_content, original_filename, output_filepath):
    global compliance_log
    target_log = compliance_log
    pdf, font_loaded = setup_pdf_with_fonts(target_log)
    pdf.add_page()
    try:
        pdf.set_font("NotoSans", "BI", 18) if font_loaded else pdf.set_font("Helvetica", "B", 18)
    except:
        pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, f"Compliance Analysis for: {original_filename}", align='C')
    pdf.ln(10)

    aspect_sections = re.split(r'(\n\*\*Compliance Aspect:.*? \*\*)\n', llm_response_content, flags=re.DOTALL)

    if aspect_sections and aspect_sections[0].strip():
        try:
            pdf.set_font("NotoSans", "", 10) if font_loaded else pdf.set_font("Helvetica", "", 10)
        except:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, aspect_sections[0].strip())

    for i in range(1, len(aspect_sections), 2):
        header = aspect_sections[i].strip()
        content = aspect_sections[i+1].strip() if i+1 < len(aspect_sections) else ""

        pdf.ln(4)
        try:
            pdf.set_font("NotoSans", "B", 11) if font_loaded else pdf.set_font("Helvetica", "B", 11)
        except:
            pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, header)
        pdf.ln(2)

        try:
            pdf.set_font("NotoSans", "", 10) if font_loaded else pdf.set_font("Helvetica", "", 10)
        except:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, content)
        pdf.ln(4)

    pdf.ln(5)

    try:
        pdf.output(output_filepath)
    except Exception as e:
        log(f"Error saving individual PDF for {original_filename} to {output_filepath}: {e}\n", target_log)

# --- Used by the section synthesis
def write_synthesis_report_to_pdf(synthesized_content, original_section_name, output_filepath):
    global section_synthesis_log
    target_log = section_synthesis_log
    pdf, font_loaded = setup_pdf_with_fonts(target_log)
    pdf.add_page()

    try:
        pdf.set_font("NotoSans", "BI", 20) if font_loaded else pdf.set_font("Helvetica", "B", 20)
    except:
        pdf.set_font("Helvetica", "B", 20)
    
    pdf.multi_cell(0, 12, f"Synthesized Section Report for: {original_section_name}", align='C')
    pdf.ln(15)

    try:
        pdf.set_font("NotoSans", "", 11) if font_loaded else pdf.set_font("Helvetica", "", 11)
    except:
        pdf.set_font("Helvetica", "", 11)
        
    pdf.multi_cell(0, 7, synthesized_content) # Directly write the synthesized content
    pdf.ln(10)

    try:
        pdf.set_font("NotoSans", "I", 9) if font_loaded else pdf.set_font("Helvetica", "I", 9)
    except:
        pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align='C')

    pdf.output(output_filepath)
    log(f"Synthesis report saved to: {output_filepath}\n", target_log)


# --------- GUI + PROGRAM FUNCTIONS ---------

# This function disables all buttons in the GUI to prevent multiple clicks during processing.
def disable_all_buttons():
    if start_editing_button: start_editing_button.config(state=tk.DISABLED)
    if section_button: section_button.config(state=tk.DISABLED)
    if sample_button: sample_button.config(state=tk.DISABLED)
    if start_synthesis_button: start_synthesis_button.config(state=tk.DISABLED)
    if guidelines_path_label: guidelines_path_label.config(state=tk.DISABLED)
    if compliance_manual_label: compliance_manual_label.config(state=tk.DISABLED)
    if compliance_aspects_input: compliance_aspects_input.config(state=tk.DISABLED)
    if guidelines_button: guidelines_button.config(state=tk.DISABLED)
    if manual_compliance_button: manual_compliance_button.config(state=tk.DISABLED)
    if editing_manual_path_label: editing_manual_path_label.config(state=tk.DISABLED)
    if section_path_label: section_path_label.config(state=tk.DISABLED)
    if start_compliance_button: start_compliance_button.config(state=tk.DISABLED)
    if guidelines_button: guidelines_button.config(state=tk.DISABLED)
    if check_button: check_button.config(state=tk.DISABLED)

def enable_all_buttons():
    if start_editing_button: start_editing_button.config(state=tk.NORMAL)
    if section_button: section_button.config(state=tk.NORMAL)
    if sample_button: sample_button.config(state=tk.NORMAL)
    if start_synthesis_button: start_synthesis_button.config(state=tk.NORMAL)
    if guidelines_path_label: guidelines_path_label.config(state=tk.NORMAL)
    if compliance_manual_label: compliance_manual_label.config(state=tk.NORMAL)
    if compliance_aspects_input: compliance_aspects_input.config(state=tk.NORMAL)
    if guidelines_button: guidelines_button.config(state=tk.NORMAL)
    if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
    if editing_manual_path_label: editing_manual_path_label.config(state=tk.NORMAL)
    if section_path_label: section_path_label.config(state=tk.NORMAL)
    if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
    if guidelines_button: guidelines_button.config(state=tk.NORMAL)
    if check_button: check_button.config(state=tk.NORMAL)

# This function initializes the LLM and embeddings instances.
#     It checks if they are already initialized and creates them if not.
def initialize_llm_and_embeddings(temp, target_log):
    import traceback  # Add this import at the beginning of the file, if it's not already there
    global llm, embeddings, compliance_log, editor_log
    try:
        if llm is None:
            log(f"Initializing LLM with LM Studio from: {LM_STUDIO_BASE_URL}\n", target_log)
            llm = ChatOpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", model="local-model", temperature=temp)
        if embeddings is None:
            log("Initializing Embeddings with CPU-Based HuggingFaceEmbeddings.\n", target_log)
            from sentence_transformers import SentenceTransformer, util  # Ensure sentence_transformers is imported here
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        return True
    except Exception as e:
        error_message = f"Error initializing LLM/Embeddings: {type(e).__name__}: {e}\n"  # Include exception type
        full_traceback = traceback.format_exc()

        # Option B: Write to a log file (most reliable for bundled apps)
        try:
            # This log file will appear in the same directory as main_app.exe (dist\main_app)
            with open("llm_embeddings_debug_error.log", "w", encoding="utf-8") as f:
                f.write(error_message + "\n\n")
                f.write("--- FULL TRACEBACK START ---\n")
                f.write(full_traceback)
                f.write("--- FULL TRACEBACK END ---\n")
            log("Full traceback saved to llm_embeddings_debug_error.log\n", target_log)  # Use log function to write to your app's log
        except Exception as file_e:
            log(f"Failed to write traceback to file: {file_e}\n", target_log)  # Use log function to write to your app's log

        messagebox.showerror("Initialization Error", f"Failed to initialize LLM or Embeddings. Ensure LM Studio is running and accessible.\nError: {e}")
        return False

# This function handles the selection of a manual document for editing.
#     It opens a file dialog to allow the user to choose a document file.
def select_editing_manual_file():
    global selected_editing_manual_path, editor_log
    target_log = editor_log
    filepath = filedialog.askopenfilename(
        title="Select Manual Document to Edit",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_editing_manual_path = filepath
        editing_manual_path_label.config(text=f"File: {os.path.basename(filepath)}")
        log(f"Selected Manual for Editing: {filepath}\n", target_log)

# This function runs the word editor logic in a separate thread to keep the GUI responsive.
#     It disables the buttons to prevent multiple clicks and checks if the necessary inputs are provided.
def run_word_editor_threaded():
    global editor_log
    target_log = editor_log
    disable_all_buttons()  # Disable buttons to prevent multiple clicks
    if not selected_editing_manual_path:
        messagebox.showwarning("Missing Input", "Please select a Document to edit.")
        if start_editing_button: start_editing_button.config(state=tk.NORMAL)
        if editing_manual_path_label: editing_manual_path_label.config(state=tk.NORMAL)
        return

    if not initialize_llm_and_embeddings(EDITING_TEMPERATURE, target_log):
        if start_editing_button: start_editing_button.config(state=tk.NORMAL)
        if editing_manual_path_label: editing_manual_path_label.config(state=tk.NORMAL)
        return

    if editor_log: editor_log.delete(1.0, tk.END) # Clear previous logs in the main log area

    import threading
    editing_thread = threading.Thread(target=perform_word_editing_logic)
    editing_thread.start()

# This function performs the main logic for the word editor.
def perform_word_editing_logic():
    global llm, editor_log, selected_editing_manual_path
    target_log = editor_log

    try:
        log("Starting document editing process...\n", target_log)

        # load the fonts for the PDF output
        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            log(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.\n", target_log)
        else:
            required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
            for font_file in required_fonts:
                if not os.path.exists(os.path.join(font_dir, font_file)):
                    log(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.\n", target_log)

        compiled_reports_list = [] # create output

        # --- Process the files given by the user ---

        full_filepath_to_edit = selected_editing_manual_path 

        if not os.path.exists(full_filepath_to_edit):
            messagebox.showerror("File Not Found", f"Error: file to edit '{os.path.basename(full_filepath_to_edit)}' not found. Please ensure it exists. Exiting.")
            log(f"Error: file to edit '{os.path.basename(full_filepath_to_edit)}' not found. Exiting.\n", target_log)
            return

        # Load the entire document as a list of LangChain Document objects
        raw_document_pages, original_document_name = load_document_content(full_filepath_to_edit, editor_log)

        # Initialize the RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=WORD_EDITOR_CHUNK_SIZE,
            length_function=len,
        )

        # Split the loaded document pages into chunks
        document_chunks = text_splitter.split_documents(raw_document_pages)
        log(f"Document split into {len(document_chunks)} chunks.\n", target_log)

        output_directory_base = "./AI_Tool_Editing_Results" # Base directory for all output files
        os.makedirs(output_directory_base, exist_ok=True)

        individual_reports_dir = os.path.join(output_directory_base, f"{os.path.splitext(original_document_name)[0]}_edited_chunks")
        os.makedirs(individual_reports_dir, exist_ok=True)
        log(f"Individual edited chunks will be saved in: {individual_reports_dir}\n", target_log)

        full_edited_content_for_display = []
        # Process each generated chunk
        for i, chunk_doc in enumerate(document_chunks):  # Iterate over LangChain Document objects
            chunk_identifier = i + 1 # Start numbering from 1 (ik it's terrible but it make sense to the user)
            log(f"\n--- Starting Analysis for Chunk {chunk_identifier} ({len(chunk_doc.page_content)} characters) ---\n", target_log)

            # Pass the page_content of the chunk_doc to the analysis function
            analysis_result = editor_analyze_single_chunk(chunk_doc.page_content, chunk_identifier, llm)
            compiled_reports_list.append(analysis_result)

            # Save individual edited chunk to PDF
            individual_pdf_output_filename = os.path.join(individual_reports_dir, f"edited_chunk_{chunk_identifier}.pdf")

            write_edited_individual_pdf(
                analysis_result["llm_response"],
                analysis_result["chunk_id"],
                original_document_name,
                individual_pdf_output_filename
            )
            log(f"Saved individual edited content for '{analysis_result['chunk_id']}' to: {individual_pdf_output_filename}\n", target_log)
            log(f"\n{'='*20} End of Analysis for {analysis_result['chunk_id']} {'='*20}\n", target_log)

        log("\n--- Generating Consolidated Edited Report ---\n", target_log)
        
        # Save the consolidated edited document report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_edited_{os.path.splitext(original_document_name)[0]}_{timestamp}.pdf")
        try:
            write_edited_consolidated_pdf(compiled_reports_list, original_document_name, consolidated_pdf_filename)
            log(f"Successfully saved consolidated edited report to PDF: {consolidated_pdf_filename}\n", target_log)
            messagebox.showinfo("Process Complete", f"Document editing finished!\nReports saved to:\n{os.path.abspath(output_directory_base)}")
        except Exception as e:
            log(f"Error saving consolidated PDF report: {e}\n", target_log)
            messagebox.showerror("Error Saving Report", f"Could not save consolidated report.\nError: {e}")

    except Exception as e:
        log(f"An unexpected error occurred during document editing: {e}\n",target_log)
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")
    finally:
        log("\nWord Editor Program Finished.\n",target_log)
        enable_all_buttons()

# This function handles the selection of a guidelines document for compliance checking.
#   It opens a file dialog to allow the user to choose a document file.
def select_guidelines_file():
    global selected_guidelines_path, compliance_log
    target_log = compliance_log
    filepath = filedialog.askopenfilename(
        title="Select Guidelines Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_guidelines_path = filepath
        guidelines_path_label.config(text=f"Guidelines: {os.path.basename(filepath)}")
        log( f"Selected Guidelines: {filepath}\n", target_log)

# This function handles the selection of a compliance manual document for compliance checking.
#   It opens a file dialog to allow the user to choose a document file.
def select_manual_files():
    global selected_compliance_manual_path, compliance_log
    target_log = compliance_log
    filepath = filedialog.askopenfilename(
        title="Select Manual Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_compliance_manual_path=filepath
        compliance_manual_label.config(text=f"Manual: {os.path.basename(filepath)}")
        log(f"Selected Manual File: {filepath}\n", target_log)

# This function runs the compliance check logic in a separate thread to keep the GUI responsive.
#   It disables the buttons to prevent multiple clicks and checks if the necessary inputs are provided.
def run_compliance_check_threaded():
    global compliance_log
    disable_all_buttons()  # Disable all buttons during compliance check
    target_log = compliance_log

    # Get compliance aspects from user input
    user_aspects_raw = compliance_aspects_input.get(1.0, tk.END).strip()
    if not user_aspects_raw:
        messagebox.showwarning("Missing Input", "Please enter compliance aspects to check.")
        if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
        if guidelines_button: guidelines_button.config(state=tk.NORMAL)
        if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
        return

    compliance_aspects_from_user = [aspect.strip() for aspect in user_aspects_raw.split('\n') if aspect.strip()]

    if not compliance_aspects_from_user:
        messagebox.showwarning("Missing Input", "Please enter valid compliance aspects (one per line).")
        if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
        if guidelines_button: guidelines_button.config(state=tk.NORMAL)
        if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
        return

    if compliance_log: compliance_log.delete(1.0, tk.END) # Clear previous logs

    if not selected_guidelines_path:
        messagebox.showwarning("Missing Input", "Please select a Guidelines Document.")
        if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
        if guidelines_button: guidelines_button.config(state=tk.NORMAL)
        if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
        return

    if not selected_compliance_manual_path:
        messagebox.showwarning("Missing Input", "Please select at least one Manual Document.")
        if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
        if guidelines_button: guidelines_button.config(state=tk.NORMAL)
        if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
        return

    if not initialize_llm_and_embeddings(COMPLIANCE_TEMPERATURE, target_log):
        if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
        if guidelines_button: guidelines_button.config(state=tk.NORMAL)
        if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
        return

    import threading
    compliance_thread = threading.Thread(target=perform_compliance_logic, args=(compliance_aspects_from_user,))
    compliance_thread.start()

# This function performs the main logic for compliance checking.
def perform_compliance_logic(user_defined_compliance_aspects):
    global llm, embeddings, compliance_log
    target_log=compliance_log
    try:
        log("Starting compliance check process...\n", target_log)

        # load the fonts for the PDF output

        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            log(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.\n", target_log)
        else:
            required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
            for font_file in required_fonts:
                if not os.path.exists(os.path.join(font_dir, font_file)):
                    log(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.\n", target_log)

        # --- 1. Load and Process Guidelines ---
        log(f"Loading guidelines from: {selected_guidelines_path}\n", target_log)
        raw_guidelines_docs, _ = load_document_content(selected_guidelines_path, compliance_log)
        if not raw_guidelines_docs:
            log(f"Error: No content loaded from guidelines document '{os.path.basename(selected_guidelines_path)}'. Aborting.\n", target_log)
            return

        # --- 2. Chunk the guidelines for the vector store ---
        guidelines_chunks = split_documents_into_chunks(raw_guidelines_docs, chunk_size=700, chunk_overlap=150)
        log("Creating Vector Store For Guidelines Document...\n", target_log)
        guidelines_vectorstore = create_vector_store(guidelines_chunks, embeddings, compliance_log)
        log(f"Guidelines document '{os.path.basename(selected_guidelines_path)}' processed and indexed into vector store.\n", target_log)
        
        # --- 3. Setup to call the compliance checker ---
        compiled_reports_list = []
        output_directory_base = "./AI_Tool_Compliance_Results" # Base directory for output reports
        os.makedirs(output_directory_base, exist_ok=True)
        individual_reports_dir = os.path.join(output_directory_base, "individual_compliance_reports")
        os.makedirs(individual_reports_dir, exist_ok=True)
        log(f"Individual compliance reports will be saved in: {individual_reports_dir}\n", target_log)

        # --- 4. Load and process manual ---
        if not selected_compliance_manual_path:
            log("Error: No manual document selected for compliance check. Aborting.\n", target_log)
            return
        full_filepath = selected_compliance_manual_path # Take the first selected file
        manual_filename_short = os.path.basename(full_filepath)
        log(f"\n--- Processing Manual Document: {manual_filename_short} ---\n", target_log)
        raw_manual_docs, loaded_manual_filename = load_document_content(full_filepath, compliance_log)
        log(f"Full manual '{loaded_manual_filename}' loaded.\n", target_log)

        # --- 5. Split the manual into chunks for compliance checking ---
        manual_text_chunks = split_documents_into_chunks(raw_manual_docs, chunk_size=COMPLIANCE_CHECKER_CHUNK_SIZE, chunk_overlap=int(COMPLIANCE_CHECKER_CHUNK_SIZE * 0.2))
        log(f"Manual document split into {len(manual_text_chunks)} chunks using RecursiveCharacterTextSplitter (chunk size: {COMPLIANCE_CHECKER_CHUNK_SIZE}).\n", target_log)

        # --- 6. Process Each Manual Chunk ---
        for i, chunk_doc in enumerate(manual_text_chunks):
            chunk_display_name = f"{loaded_manual_filename}_chunk_{i+1}"
            log(f"\n--- Analyzing Manual Chunk {i+1}/{len(manual_text_chunks)} ({chunk_display_name}) ---\n", target_log)
            manual_chunk_full_text = chunk_doc.page_content # The content of the current chunk

            log(f"LLM: Identifying relevant aspects for {chunk_display_name}...\n", target_log)
            relevant_aspects_for_this_chunk = get_llm_relevant_aspects(
                manual_chunk_full_text,
                user_defined_compliance_aspects,
                llm
            )

            if not relevant_aspects_for_this_chunk:
                log(f"LLM found no relevant compliance aspects for '{chunk_display_name}'. Skipping compliance check for this chunk.\n", target_log)
                compliance_result_text = f"No relevant compliance aspects found for '{chunk_display_name}' by LLM filtering based on user input."
                compiled_reports_list.append({"chunk_filename": chunk_display_name, "llm_response": compliance_result_text})
                base_name = os.path.splitext(chunk_display_name)[0]
                individual_pdf_output_filename = os.path.join(individual_reports_dir, f"compliance_report_for_{base_name}.pdf")
                write_individual_response_to_pdf(
                    compliance_result_text,
                    chunk_display_name,
                    individual_pdf_output_filename
                )
                continue

            log(f"LLM identified relevant aspects for '{chunk_display_name}':\n", target_log)
            for aspect in relevant_aspects_for_this_chunk:
                log(f" - {aspect}\n", target_log)

            log(f"Starting Compliance Analysis for {chunk_display_name}...\n", target_log)

            compliance_result_text = check_manual_compliance(
                manual_chunk_full_text,
                chunk_display_name, # Pass the chunk display name
                guidelines_vectorstore,
                llm,
                relevant_aspects_for_this_chunk,
                TOP_K_GUIDELINES
            )

            compiled_reports_list.append({"chunk_filename": chunk_display_name, "llm_response": compliance_result_text})

            base_name = os.path.splitext(chunk_display_name)[0]
            individual_pdf_output_filename = os.path.join(individual_reports_dir, f"compliance_report_for_{base_name}.pdf")

            write_individual_response_to_pdf(
                compliance_result_text,
                chunk_display_name,
                individual_pdf_output_filename
            )
            log(f"Saved individual compliance report for '{chunk_display_name}' to: {individual_pdf_output_filename}\n", target_log)
            log(f"\n{'='*20} End of Analysis for '{chunk_display_name}' {'='*20}\n", target_log)

        # --- 7. Consolidated Report Generation ---
        log("\n--- Generating Consolidated Compliance Report ---\n", target_log)
        consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        try:
            write_report_to_pdf(compiled_reports_list, consolidated_pdf_filename)
            log(f"Successfully saved consolidated compliance report to PDF: {consolidated_pdf_filename}\n", target_log)
            messagebox.showinfo("Process Complete", f"Compliance check finished!\nReports saved to:\n{os.path.abspath(output_directory_base)}")
        except Exception as e:
            log(f"Error saving consolidated PDF report: {e}\n", target_log)
            messagebox.showerror("Error Saving Report", f"Could not save consolidated report.\nError: {e}")

    except Exception as e:
        log(f"An unexpected error occurred during the compliance check: {e}\n", target_log)
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")
    finally:
        # Check if the buttons exist before trying to configure them
        if 'start_compliance_button' in globals() and start_compliance_button.winfo_exists():
            start_compliance_button.config(state=tk.NORMAL)
        if 'guidelines_button' in globals() and guidelines_button.winfo_exists():
            guidelines_button.config(state=tk.NORMAL)
        if 'manual_button' in globals() and manual_compliance_button.winfo_exists(): # Assuming this is your add manual file button
            manual_compliance_button.config(state=tk.NORMAL)
        log("\nProgram Finished.\n", target_log)
        enable_all_buttons()

# This function handles the selection of a section document for synthesis.
#   It opens a file dialog to allow the user to choose a document file.
def select_section_file():
    global selected_section_path, section_path_label, section_synthesis_log 
    target_log = section_synthesis_log
    filepath = filedialog.askopenfilename(
        title="Select Source Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_section_path = filepath
        # Ensure the label widget exists before trying to update it
        if section_path_label:
            section_path_label.config(text=f"Source: {os.path.basename(filepath)}")
        log(f"Selected Source Document for Synthesis: {filepath}\n", target_log)

# This function handles the selection of a sample text document for section synthesis.
#   It opens a file dialog to allow the user to choose a document file.
def select_sample_file():
    global selected_sample_path, sample_path_label, section_synthesis_log 
    target_log = section_synthesis_log
    filepath = filedialog.askopenfilename(
        title="Select Sample Text Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_sample_path = filepath
        # Ensure the label widget exists before trying to update it
        if sample_path_label:
            sample_path_label.config(text=f"Sample: {os.path.basename(filepath)}")
        log(f"Selected Sample Text Document: {filepath}\n", target_log)

# This function runs the section synthesis logic in a separate thread to keep the GUI responsive.
def run_synthesis_threaded():
    global section_synthesis_log
    disable_all_buttons() # Disable all relevant buttons across the app
    target_log = section_synthesis_log

    if not selected_section_path:
        messagebox.showwarning("Missing Input", "Please select a Source Document to synthesize.")
        # Re-enable specific buttons for this tab on error
        if start_synthesis_button: start_synthesis_button.config(state=tk.NORMAL)
        if section_button: section_button.config(state=tk.NORMAL)
        if sample_button: sample_button.config(state=tk.NORMAL)
        return

    if not selected_sample_path:
        messagebox.showwarning("Missing Input", "Please select a Sample Text Document for style match.")
        # Re-enable specific buttons for this tab on error
        if start_synthesis_button: start_synthesis_button.config(state=tk.NORMAL)
        if section_button: section_button.config(state=tk.NORMAL)
        if sample_button: sample_button.config(state=tk.NORMAL)
        return

    if not initialize_llm_and_embeddings(EDITING_TEMPERATURE, target_log):
        if start_synthesis_button: start_synthesis_button.config(state=tk.NORMAL)
        if section_button: section_button.config(state=tk.NORMAL)
        if sample_button: sample_button.config(state=tk.NORMAL)
        return

    # Clear the log for this tab specifically
    if section_synthesis_log: section_synthesis_log.delete(1.0, tk.END)

    import threading
    synthesis_thread = threading.Thread(target=perform_section_synthesis_logic)
    synthesis_thread.start()

# This function performs the main logic for section synthesis.
def perform_section_synthesis_logic():
    global llm, section_synthesis_log, selected_section_path, selected_sample_path
    target_log = section_synthesis_log
    try:
        log(f"Starting section synthesis process...\n", target_log)

        # --- font setup ---
        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            log(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.\n", target_log)
        else:
            required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
            for font_file in required_fonts:
                if not os.path.exists(os.path.join(font_dir, font_file)):
                    log(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.\n", target_log)

        compiled_reports_list = []
        output_directory_base = "./AI_Tool_Section_Synthesis_Results"
        os.makedirs(output_directory_base, exist_ok=True)
        log(f"Output directory created: {output_directory_base}\n", target_log)

        # --- Load Source Section Document ---
        if not selected_section_path:
            log(f"Error: No source document selected for synthesis. Aborting.\n", target_log)
            messagebox.showwarning("Missing Input", "Please select a Source Document for synthesis.")
            return

        log(f"Loading source section from: {selected_section_path}\n", target_log)
        raw_section_docs, section_filename_short = load_document_content(selected_section_path, section_synthesis_log)
        if not raw_section_docs:
            log(f"Error: No content loaded from source document '{section_filename_short}'. Aborting.\n", target_log)
            messagebox.showerror("Loading Error", f"Could not load content from source document: {section_filename_short}")
            return

        full_section_text = "\n\n".join([doc.page_content for doc in raw_section_docs])
        log(f"Full source section '{section_filename_short}' loaded with {len(full_section_text)} characters.\n", target_log)

        # --- Load Sample Text Document ---
        if not selected_sample_path:
            log(f"Error: No sample document selected for style match. Aborting.\n", target_log)
            messagebox.showwarning("Missing Input", "Please select a Sample Text Document for style match.")
            return

        log(f"Loading sample text from: {selected_sample_path}\n", target_log)
        raw_sample_docs, sample_filename_short = load_document_content(selected_sample_path, section_synthesis_log)
        if not raw_sample_docs:
            log(f"Error: No content loaded from sample document '{sample_filename_short}'. Aborting.\n", target_log)
            messagebox.showerror("Loading Error", f"Could not load content from sample document: {sample_filename_short}")
            return

        full_sample_text = "\n\n".join([doc.page_content for doc in raw_sample_docs])
        log(f"Full sample text '{sample_filename_short}' loaded with {len(full_sample_text)} characters.\n", target_log)

        # --- Perform the Synthesis ---
        log(f"Calling LLM for section synthesis...\n", target_log)
        analysis_result = section_synthesis(full_section_text, full_sample_text, llm)
        compiled_reports_list.append({"section_filename": section_filename_short, "synthesis_result": analysis_result})

        # --- Write report to PDF ---
        log("\n--- Generating Consolidated Synthesis Report ---\n", target_log)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Use the name of the section file for the output PDF
        output_base_name = os.path.splitext(section_filename_short)[0]
        consolidated_pdf_filename = os.path.join(output_directory_base, f"synthesized_report_for_{output_base_name}_{timestamp}.pdf")
        try:
            write_synthesis_report_to_pdf(analysis_result, section_filename_short, consolidated_pdf_filename)
            log(f"Successfully saved synthesized section to PDF: {consolidated_pdf_filename}\n", target_log)
            messagebox.showinfo("Process Complete", f"Section Synthesis finished!\nOutput saved to:\n{os.path.abspath(output_directory_base)}")
        except Exception as e:
            log(f"Error saving consolidated PDF report: {e}\n", target_log)
            messagebox.showerror("Error Saving Report", f"Could not save consolidated report.\nError: {e}")

    except Exception as e:
        log(f"An unexpected error occurred during section synthesis: {e}\n", target_log)
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")
    finally:
        # Re-enable buttons specific to this tab
        if start_synthesis_button and start_synthesis_button.winfo_exists(): start_synthesis_button.config(state=tk.NORMAL)
        if section_button and section_button.winfo_exists(): section_button.config(state=tk.NORMAL)
        if sample_button and sample_button.winfo_exists(): sample_button.config(state=tk.NORMAL)
        log("\nSection Synthesis Program Finished.\n", target_log)
        enable_all_buttons()

def diagnostics_logic():
    global troubleshooting_log
    target_log = troubleshooting_log
    log("Running diagnostic checks...\n", target_log)
    log("Checking LLM (LM Studio) connection...", target_log)
    try:
        temp_llm = ChatOpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", model="local-model", temperature=0)
        # A simple invoke to test connectivity
        test_response = temp_llm.invoke("give me 3 words, one of which is 'test'")
        if "test" in test_response.content.lower():
            log("LLM connection successful! Model is responding.\n", target_log)
        else:
            log(f"LLM connected but response was unexpected: '{test_response.content}'. Check model loading in LM Studio.\n", target_log)
    except Exception as e:
        log(f"LLM connection FAILED: {e}. Ensure LM Studio is running and a model is loaded.\n", target_log)
    # Check font files
    log("Checking custom font files...\n", target_log)
    
    # *** FIX HERE: Use resource_path for the fonts directory ***
    font_dir = resource_path("fonts") 

    required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
    
    found_all_fonts = True
    if not os.path.exists(font_dir):
        log(f"Font directory '{font_dir}' NOT found.", target_log)
        found_all_fonts = False
    else:
        log(f"Font directory '{font_dir}' found.", target_log)
        for font_file in required_fonts:
            font_path = os.path.join(font_dir, font_file)
            if not os.path.exists(font_path):
                log(f"Required font file '{font_file}' NOT found in '{font_dir}'.", target_log)
                found_all_fonts = False
            else:
                log(f"Font file '{font_file}' found.", target_log)
    
    if found_all_fonts:
        log("All required font files are present.\n", target_log)
    else:
        log("Some font files are missing or the 'fonts' directory is not found. PDF generation might use default fonts or fail.\n", target_log)

    log("Diagnostic checks complete.", target_log)
    enable_all_buttons()

def run_diagnostics_threaded():
    disable_all_buttons()
    troubleshooting_log.delete(1.0, tk.END)

    import threading
    diagnostic_thread = threading.Thread(target=diagnostics_logic)
    diagnostic_thread.start()
        
# --- GUI Setup ---

# Functions to initialize the main app components (notebook and tabs)

def create_compliance_checker_tab_frame(parent):
    global compliance_log, guidelines_path_label, guidelines_button, compliance_manual_label, manual_compliance_button, compliance_aspects_input, start_compliance_button 
    target_log = compliance_log
    compliance_tab_frame = ttk.Frame(parent, padding="10")

    instructions_text = """
    This tab helps you assess the compliance of your "Manual Documents" against a "Guidelines Document" based on specific "Compliance Aspects".

    1. If you have not already, follow the instructions on the 'Home' screen to start LM Studio (can also be found at the tab on the top left).
    2. Select the 'Guidelines Document' (e.g., regulations, policies).
    3. Select the 'Manual Documents' (the document you want to check for compliance).
    4. Review and edit the 'Compliance Aspects' list to focus the LLM's analysis. For example, if your manual section is about Employee Contributions,
            "Compliance Aspects" should contain "Employee Contributions".
    5. Click 'Start Compliance Check' to generate a detailed compliance report that will be saved as a PDF on your computer.
    
    IMPORTANT NOTE: For longer 'Manual Documents' or extensive 'Compliance Aspects' lists, the process may take significant time.
    You can run it in the background, but ensure both LM Studio and this application remain open.
    
    IMPORTANT NOTE: The generated report may suggest additions or refinements even for largely compliant sections.
    Review the reports carefully for comprehensive insights.    
    """
    instruction_label = ttk.Label(compliance_tab_frame,
                                  text=instructions_text.strip(),
                                  wraplength=800,
                                  justify=tk.LEFT)
    instruction_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew") 

    file_frame = ttk.LabelFrame(compliance_tab_frame, text="Document Selection", padding="10")
    file_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew") 

    ttk.Label(file_frame, text="Guidelines Document:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    guidelines_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    guidelines_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    guidelines_button = ttk.Button(file_frame, text="Browse Files", command=select_guidelines_file)
    guidelines_button.grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(file_frame, text="Manual Documents:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    compliance_manual_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    compliance_manual_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    manual_compliance_button = ttk.Button(file_frame, text="Browse Files", command=select_manual_files)
    manual_compliance_button.grid(row=1, column=2, padx=5, pady=5, sticky="nw")

    ttk.Label(file_frame, text="Compliance Aspects (one per line):").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
    compliance_aspects_input = scrolledtext.ScrolledText(file_frame, height=5, width=50, wrap=tk.WORD)
    compliance_aspects_input.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    compliance_aspects_input.insert(tk.END, """Functions, members, and procedures of the Social Security Board\nResponsibilities of the Administrator\nAudit Requirements and External Oversight\nEmployee contribution amounts and regulations\nClassification of self-employed peoples\nClassification of disabilities\nEmployer contribution amounts and regulations\nThe keeping of accounts and reports\nAppeals and Dispute Resolution Mechanisms\nEmployee offenses and penalties\nHealth insurance benefits\nExclusions and limitations of health insurance coverage\nReimbursement procedures and requirements\nSubscription and enrollment processes\nPrivacy and data protection policies\nSuccession and transfer of medical savings account after death\nDue dates""")

    file_frame.columnconfigure(1, weight=1)

    control_frame = ttk.LabelFrame(compliance_tab_frame, text="Controls and Output", padding="10")
    control_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
    def start_compliance_check_button_action():
        log(f"Starting compliance check with manual document {selected_compliance_manual_path} and guidelines from {selected_guidelines_path}.\n", target_log)
        run_compliance_check_threaded()

    start_compliance_button = ttk.Button(control_frame, text="Start Compliance Check", command=start_compliance_check_button_action)
    start_compliance_button.pack(pady=10)

    log_frame = ttk.LabelFrame(control_frame, text="Program Information Log", padding="5")
    log_frame.pack(expand=True, fill="both", padx=5, pady=5)

    compliance_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    compliance_log.pack(expand=True, fill="both")
    compliance_log.see(tk.END)

    compliance_tab_frame.grid_columnconfigure(0, weight=1)
    compliance_tab_frame.grid_columnconfigure(1, weight=1)

    compliance_tab_frame.grid_rowconfigure(0, weight=0) 
    compliance_tab_frame.grid_rowconfigure(1, weight=0) 
    compliance_tab_frame.grid_rowconfigure(2, weight=1) 
    return compliance_tab_frame

def create_word_editor_tab_frame(parent):
    global editor_log, start_editing_button, editing_manual_path_label
    target_log = editor_log
    word_editor_tab_frame = ttk.Frame(parent, padding="10")

    instruction_text = """
    This tab uses AI to enhance the clarity, correct grammar, fix typos, and improve the wording of your selected document.
    
        1. If you have not already, follow the instructions on the 'Home' screen to start LM Studio (can also be found at the tab on the top left).
        2. Select the 'Document to Edit' that you wish to refine.
        3. Click 'Start AI Editing' to begin the process.
        4. The edited content will appear in a PDF on your computer.
    
    IMPORTANT NOTE: This program processes your document in sections. Please review the transitions between these sections carefully, as issues may occur where the chunks connect.
    """
    instruction_label = ttk.Label(word_editor_tab_frame,
                                  text=instruction_text.strip(),
                                  wraplength=800,
                                  justify=tk.LEFT)
    instruction_label.pack(pady=10, padx=10, fill="x")

    file_frame = ttk.LabelFrame(word_editor_tab_frame, text="Select Document for Editing", padding="10")
    file_frame.pack(pady=10, padx=10, fill="x")

    ttk.Label(file_frame, text="Document to Edit:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    editing_manual_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    editing_manual_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(file_frame, text="Browse File", command=select_editing_manual_file).grid(row=0, column=2, padx=5, pady=5)
    file_frame.columnconfigure(1, weight=1)

    control_frame = ttk.LabelFrame(word_editor_tab_frame, text="Controls", padding="10")
    control_frame.pack(pady=10, padx=10, fill="x")

    def start_editing_button_action():
        log(f"Starting AI editing for document: {selected_editing_manual_path}\n", target_log)
        run_word_editor_threaded()
    start_editing_button = ttk.Button(control_frame, text="Start AI Editing", command=start_editing_button_action)
    start_editing_button.pack(pady=10)

    log_frame = ttk.LabelFrame(control_frame, text="Program Information Log", padding="5")
    log_frame.pack(expand=True, fill="both", padx=5, pady=5)

    editor_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    editor_log.pack(expand=True, fill="both")
    editor_log.see(tk.END)

    word_editor_tab_frame.grid_columnconfigure(0, weight=1)
    word_editor_tab_frame.grid_columnconfigure(1, weight=1)
    word_editor_tab_frame.grid_rowconfigure(1, weight=1)

    return word_editor_tab_frame

def create_troubleshooting_tab_frame(parent):
    global troubleshooting_log, troubleshooting_tab_frame, check_button

    troubleshooting_tab_frame = ttk.Frame(parent, padding="10")

    # --- Introduction/Purpose ---
    intro_label = ttk.Label(troubleshooting_tab_frame, 
                            text="This tab provides guidance for common issues and allows you to perform basic checks. Before proceeding, follow the instructions on the 'Home' screen to start LM Studio (can also be found at the tab on the top left).", wraplength=600, justify=tk.LEFT)
    intro_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    # --- Common Issues Section ---
    issues_frame = ttk.LabelFrame(troubleshooting_tab_frame, text="Common Issues & Solutions", padding="10")
    issues_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
    troubleshooting_tab_frame.grid_rowconfigure(1, weight=1)

    # Issue 1: LLM (LM Studio) Connection
    ttk.Label(issues_frame, text="1. LLM (LM Studio) Connection Issues:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(issues_frame, text="- Ensure LM Studio is running and a model is loaded.\n- Verify the 'Server URL' in LM Studio settings (must be http://localhost:1234).\n- Check firewall settings if connection is blocked.", wraplength=550, justify=tk.LEFT).grid(row=1, column=0, sticky="w", padx=15, pady=2)
    
    # Issue 2: File Loading/Access
    ttk.Label(issues_frame, text="2. Document Loading/Access Errors:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(issues_frame, text="- Ensure input files (.pdf, .docx, .txt) are not corrupted or password-protected.\n- Check file permissions to ensure the application can read them.\n- Avoid extremely large files if experiencing memory issues.", wraplength=550, justify=tk.LEFT).grid(row=5, column=0, sticky="w", padx=15, pady=2)

    # Issue 3: Program is taking a long time
    ttk.Label(issues_frame, text="3. Program is taking a long time:").grid(row=7, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(issues_frame, text="- Check the Performance tab in 'Task Manager'. If CPU usage is high and GPU usage is low, do the following. \n   - In LM Studio navigate to 'Developer' tab (select on the left)\n    - Click the 'Load' tab on the right hand side\n    - Slide the 'GPU Offload' further to the right to increase GPU usage.\n- Other fixes:\n    - Avoid very large files\n    - Close other running applications", wraplength=550, justify=tk.LEFT).grid(row=8, column=0, sticky="", padx=25, pady=2)

    # --- Diagnostic Checks Section ---
    diagnostics_frame = ttk.LabelFrame(troubleshooting_tab_frame, text="Diagnostic Checks", padding="10")
    diagnostics_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    check_button = ttk.Button(diagnostics_frame, text="Run Diagnostics", command=run_diagnostics_threaded)
    check_button.pack(pady=5)

    # --- Diagnostic Log ---
    log_frame = ttk.LabelFrame(troubleshooting_tab_frame, text="Diagnostic Log", padding="5")
    log_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    troubleshooting_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    troubleshooting_log.pack(expand=True, fill="both")
    troubleshooting_log.insert(tk.END, "Diagnostic messages will appear here.\n")
    troubleshooting_log.see(tk.END)

    # --- Grid configuration for troubleshooting_tab_frame ---
    troubleshooting_tab_frame.grid_columnconfigure(0, weight=1)
    troubleshooting_tab_frame.grid_columnconfigure(1, weight=1)
    troubleshooting_tab_frame.grid_rowconfigure(0, weight=0) # Intro
    troubleshooting_tab_frame.grid_rowconfigure(1, weight=0) # Issues
    troubleshooting_tab_frame.grid_rowconfigure(2, weight=0) # Diagnostics controls
    troubleshooting_tab_frame.grid_rowconfigure(3, weight=1) # Diagnostic log (expands)

    return troubleshooting_tab_frame

def create_section_synthesis_tab_frame(parent):
    global section_synthesis_log, section_path_label, section_button, sample_path_label, sample_button, start_synthesis_button
    target_log = section_synthesis_log
    section_synthesis_tab_frame = ttk.Frame(parent, padding="10")

    instruction_text = """
    This tab helps you turn the information in a 'Source Document' into a statement, matching the style and organization of a provided sample text.

    IMPORTANT NOTE: This tool works best when the documents provided are short. 
    Aim for a 'Source Document' ~ 3500 characters & 'Sample Document' ~1000 characters.

    1. If you have not already, follow the instructions on the 'Home' screen to start LM Studio (can also be found at the tab on the top left).
    2. Select the 'Source Document' containing the section you wish to synthesize.
    3. Select a 'Sample Text Document' to guide the AI's writing style.
    4. Click 'Start Synthesis' to generate the new content. Results will be saved as a PDF on your computer
    """
    instruction_label = ttk.Label(section_synthesis_tab_frame,
                                  text=instruction_text.strip(),
                                  wraplength=800,
                                  justify=tk.LEFT)
    instruction_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

    file_frame = ttk.LabelFrame(section_synthesis_tab_frame, text="Document Selection", padding="10")
    file_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

    ttk.Label(file_frame, text="Source Document:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    section_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    section_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    section_button = ttk.Button(file_frame, text="Browse Source", command=select_section_file)
    section_button.grid(row=0, column=2, padx=5, pady=5)

    ttk.Label(file_frame, text="Sample Text Document:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    sample_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    sample_path_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    sample_button = ttk.Button(file_frame, text="Browse Sample", command=select_sample_file)
    sample_button.grid(row=1, column=2, padx=5, pady=5)
    file_frame.columnconfigure(1, weight=1)

    control_frame = ttk.LabelFrame(section_synthesis_tab_frame, text="Controls", padding="10")
    control_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

    def start_synthesis_button_action():
        log(f"Starting section synthesis with source document: {selected_section_path} and sample text: {selected_sample_path}\n", section_synthesis_log)
        run_synthesis_threaded()

    start_synthesis_button = ttk.Button(control_frame, text="Start Synthesis", command=start_synthesis_button_action)
    start_synthesis_button.pack(pady=10)

    log_frame = ttk.LabelFrame(section_synthesis_tab_frame, text="Program Information Log", padding="5")
    log_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    section_synthesis_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    section_synthesis_log.pack(expand=True, fill="both")
    section_synthesis_log.insert(tk.END, "Section Synthesis Log Initialized.\n")
    section_synthesis_log.see(tk.END)

    section_synthesis_tab_frame.grid_columnconfigure(0, weight=1)
    section_synthesis_tab_frame.grid_columnconfigure(1, weight=1)
    section_synthesis_tab_frame.grid_columnconfigure(2, weight=0) 
    section_synthesis_tab_frame.grid_rowconfigure(0, weight=0) 
    section_synthesis_tab_frame.grid_rowconfigure(1, weight=0)
    section_synthesis_tab_frame.grid_rowconfigure(2, weight=0) 
    section_synthesis_tab_frame.grid_rowconfigure(3, weight=1) 

    return section_synthesis_tab_frame

def initialize_main_app_components():
    global notebook, compliance_tab_frame, word_editor_tab_frame, troubleshooting_tab_frame, tab_frames, home_frame

    if notebook is None:
        notebook = ttk.Notebook(main_app_frame)
        notebook.pack(expand=True, fill="both")

        home_tab_frame = create_home_screen(notebook)
        notebook.add(home_tab_frame, text="Home")
        tab_frames["Home"] = home_tab_frame

        compliance_tab_frame = create_compliance_checker_tab_frame(notebook)
        word_editor_tab_frame = create_word_editor_tab_frame(notebook)
        section_synthesis_tab_frame = create_section_synthesis_tab_frame(notebook)
        troubleshooting_tab_frame = create_troubleshooting_tab_frame(notebook)

        notebook.add(compliance_tab_frame, text="Compliance Checker")
        tab_frames["Compliance Checker"] = compliance_tab_frame

        notebook.add(word_editor_tab_frame, text="Word Editor")
        tab_frames["Word Editor"] = word_editor_tab_frame

        notebook.add(section_synthesis_tab_frame, text="Section Synthesis")
        tab_frames["Section Synthesis"] = section_synthesis_tab_frame

        notebook.add(troubleshooting_tab_frame, text="Troubleshooting")
        tab_frames["Troubleshooting"] = troubleshooting_tab_frame

def show_main_app(target_tab_key):
    home_frame.pack_forget()
    main_app_frame.pack(fill="both", expand=True, padx=10, pady=10)

    initialize_main_app_components()

    if target_tab_key in tab_frames:
        notebook.select(tab_frames[target_tab_key])
    else:
        print(f"Warning: Attempted to select unknown tab: {target_tab_key}. Falling back to Home tab.")
        notebook.select(tab_frames.get("Home", 0))

def create_home_screen(parent):
    global home_frame
    home_frame = ttk.Frame(parent, padding="20")
    home_frame.pack(fill="both", expand=True)

    ttk.Label(home_frame, text="Welcome to the Advanced Document AI Assistant!",
              font=("Helvetica", 18, "bold")).pack(pady=20)

    ttk.Label(home_frame, text=(
        "Before proceeding, please ensure you have LM Studio installed and open on your computer and that you have done the following:\n\n"), justify="center", font=("Helvetica", 11)).pack(pady=10)
        
    ttk.Label(home_frame,text=("1.  In LM Studio, navigate to the 'Developer' tab on the left and select a local model from the dropdown at the top.\n"
        "     We suggest mistral-7b-instruct-v0.1 as it is a good balance of performance and speed.\n\n"
        "2.  Make sure the status is 'Running' and that the green text says 'READY'\n\n"
        "3.  LM Studio should have its local server 'Reachable at' http://localhost:1234\n\n"
        "4.  In the 'load' tab on the right, adjust the 'GPU Offload' scale to be higher for improved performance.\n" 
        "     Note that this will use more GPU memory, so ensure you have enough available. If not, lower GPU Offload may be faster\n\n"
    ), wraplength=800, justify="left", font=("Helvetica", 11)).pack(pady=10)

    ttk.Label(home_frame, text=(
        "\nChoose your application:"
    ), font=("Helvetica", 12, "italic")).pack(pady=20)

    # Buttons for specific tab navigation, now passing the key for the dictionary lookup
    compliance_button = ttk.Button(home_frame, text="Continue to Compliance Checker",
                                   command=lambda: show_main_app("Compliance Checker"))
    compliance_button.pack(pady=5)

    word_editor_button = ttk.Button(home_frame, text="Continue to Word Editor",
                                    command=lambda: show_main_app("Word Editor"))
    word_editor_button.pack(pady=5)

    section_synthesis_button = ttk.Button(home_frame, text="Continue to Section Synthesis",
                                    command=lambda: show_main_app("Section Synthesis"))
    section_synthesis_button.pack(pady=5)

    return home_frame

# --- MAIN GUI WINDOW ---
root = tk.Tk()
root.title("ROPSSA Operations Manual Tool")
root.geometry("1000x750") # Initial size for the home screen

# Create a container frame for the main application, initially hidden
main_app_frame = ttk.Frame(root)
# This will be packed later by show_main_app

# Create and display the home screen first
home_frame = create_home_screen(root)

# Run the application
root.mainloop()