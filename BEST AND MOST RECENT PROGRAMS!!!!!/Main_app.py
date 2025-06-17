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

def log(text):
    if compliance_log is not None:
        compliance_log.insert(tk.END, text + "\n")
        compliance_log.see(tk.END)
    elif editor_log is not None:
        editor_log.insert(tk.END, text + "\n")
        editor_log.see(tk.END)
    if section_synthesis_log is not None:
        section_synthesis_log.insert(tk.END, text + "\n")
        section_synthesis_log.see(tk.END)

# --- CORE FUNCTIONS FOR COMPLIANCE CHECKER ---

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

    log(f"Loading raw content from {os.path.basename(filepath)}...\n")
    try:
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["source"] = os.path.basename(filepath)
        return raw_docs, os.path.basename(filepath)
    except Exception as e:
        log( f"Error loading {os.path.basename(filepath)}: {e}\n") 
        return [], None

def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks, embeddings_model):
    log("Creating vector store from document chunks...\n")
    return FAISS.from_documents(chunks, embeddings_model)

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

def get_llm_relevant_aspects(manual_chunk_content, available_aspects, llm_model):
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
    4.  **List Length:** Keep the list short and high-level (aim for 5-10 items) but allow items to conceptually encapsulate more.
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
        log( "   LLM: Identifying relevant aspects for chunk...\n")
        start_time = time.time()
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        log(f"   LLM aspect selection took {elapsed:.2f} seconds.\n")
        selected_aspects_raw = response.content.strip()

        if selected_aspects_raw.upper() == "NONE" or not selected_aspects_raw:
            return []

        selected_aspects = [aspect.strip() for aspect in selected_aspects_raw.split('\n')]
        return selected_aspects
    except Exception as e:
        log(f"Error during LLM aspect selection for chunk: {e}\n")
        return []

def check_manual_compliance(manual_chunk_full_text, manual_chunk_index, guidelines_vectorstore, llm_model, compliance_aspects_for_this_chunk, top_k_value):
    log(f"\n--- Checking Compliance for Document Chunk: '{manual_chunk_index}' ---\n")
    compliance_report_content = []
    current_date = date.today().strftime("%B %d, %Y")

    if not compliance_aspects_for_this_chunk:
        compliance_report_content.append("No specific compliance aspects were identified as relevant to this chunk by the LLM. Cannot perform targeted checks.")
        log("No specific compliance aspects found as relevant by LLM.\n")
        return "\n".join(compliance_report_content)

    for aspect in compliance_aspects_for_this_chunk:
        log(f"   Checking aspect: '{aspect}'...\n")

        log(f"    Retrieving relevant guidelines for aspect: '{aspect}'...\n")
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
            log(f"    Prompting LLM for compliance analysis on aspect: '{aspect}'...\n")
            response = llm_model.invoke(prompt)
            elapsed = time.time() - start_time
            log(f"    Compliance LLM call took {elapsed:.2f} seconds.\n")
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n{response.content.strip()}")
        except Exception as e:
            compliance_report_content.append(f"\n**Compliance Aspect: {aspect}**\n   - Error during analysis: {e}")
            log(f"   Error analyzing aspect '{aspect}' for '{manual_chunk_index}': {e}\n")

    return "\n".join(compliance_report_content)

# --- CORE FUNCTIONS FOR WORD EDITOR ---

# --- editor_analyze_single_chunk function remains largely the same ---
def editor_analyze_single_chunk(chunk_full_text, chunk_index, llm_model): 
    print(f"\n--- Analyzing Chunk {chunk_index} ---")

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
        log(f"   LLM editing took {elapsed:.2f} seconds for Chunk {chunk_index}.\n")
        # Access the content attribute of the AIMessage object
        return {"chunk_id": f"Chunk {chunk_index}", "llm_response": response.content} # MODIFIED LINE
    except Exception as e:
        log(f"Error processing 'Chunk {chunk_index}': {e}\n")
        return {"chunk_id": f"Chunk {chunk_index}", "llm_response": f"Error processing 'Chunk {chunk_index}': {e}"}

# --- Section Synthesis Functions ---
def section_synthesis(section_text, sample_text, llm_model):
    synthesis_result = ""
    log(f"\n--- Synthesizing Section... ---\n") # Explicitly target log

    if not section_text:
        synthesis_result = "No section text provided for synthesis."
        log("No section text provided for synthesis.\n")
        return synthesis_result

    if not sample_text:
        synthesis_result = "No sample text provided for style match."
        log("No sample text provided for style match.\n")
        return synthesis_result

    prompt = f"""
    You are an expert in document synthesis and analysis. Your task is to analyze the provided section of text and synthesize it into a concise policy statement in the style of the sample text.

    **Instructions:**
    1. Analyze the section text carefully.
    2. Provide a concise policy statement that captures the key points, themes, and insights from the section written with the style and organization of the sample text.
    3. Ensure that your response is clear, coherent, and well-structured.

    **Section Text:**
    {section_text}

    **Sample Text:**
    {sample_text}

    **Your Summary/Analysis:**
    """
    try:
        start_time = time.time()
        response = llm_model.invoke(prompt)
        elapsed = time.time() - start_time
        log(f"    LLM section synthesis took {elapsed:.2f} seconds.\n")
        synthesis_result = response.content.strip()
    except Exception as e:
        synthesis_result = f"Error during section synthesis: {e}"
        log(f"Error during section synthesis: {e}\n")

    return synthesis_result    

# --- PDF Writing Functions  ---

# This helper function centralizes PDF object creation and font loading
def setup_pdf_with_fonts():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_dir = "./fonts"
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
            log(f"Warning: NotoSans-Regular.ttf not found at {regular_font_path}. Using default font.\n")
        if os.path.exists(italic_font_path):
            pdf.add_font("NotoSans", "I", italic_font_path, uni=True)
        if os.path.exists(bold_font_path):
            pdf.add_font("NotoSans", "B", bold_font_path, uni=True)
        if os.path.exists(bolditalic_font_path):
            pdf.add_font("NotoSans", "BI", bolditalic_font_path, uni=True)
    except Exception as e:
        log(f"Error loading custom fonts: {e}. Falling back to default.\n")
        font_loaded = False # Ensure font_loaded is false if there's an error

    return pdf, font_loaded

# EDITOR ONES
def write_edited_consolidated_pdf(report_data, original_doc_name, output_filepath):
    pdf, font_loaded = setup_pdf_with_fonts() # Use the helper function
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
    pdf, font_loaded = setup_pdf_with_fonts() # Use the helper function
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
        log(f"Error saving individual PDF for {chunk_id} to {output_filepath}: {e}\n")

#COMPLIANCE CHECKER ONES
def write_report_to_pdf(report_data, output_filepath):
    pdf, font_loaded = setup_pdf_with_fonts()
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
    pdf, font_loaded = setup_pdf_with_fonts()
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
        log(f"Error saving individual PDF for {original_filename} to {output_filepath}: {e}\n")

# SYNTHESIS ONES
def write_synthesis_report_to_pdf(synthesized_content, original_section_name, output_filepath):
    pdf, font_loaded = setup_pdf_with_fonts()
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
    log(f"Synthesis report saved to: {output_filepath}\n")

# --- Configuration Parameters ---
TOP_K_GUIDELINES = 7
COMPLIANCE_TEMPERATURE = 0.1
EDITING_TEMPERATURE = 0.5 
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# --- GLOBAL CONSTANTS FOR WORD EDITOR ---
WORD_EDITOR_CHUNK_SIZE = 4000

# --- GLOBAL CONSTANTS FOR COMPLIANCE CHECKER ---
COMPLIANCE_CHECKER_CHUNK_SIZE = 3000

# --- GLOBAL VARIABLES FOR GUI STATE ---
selected_guidelines_path = ""
selected_compliance_manual_path = None
selected_editing_manual_path = ""
selected_section_path = ""
selected_sample_path = ""
llm = None
embeddings = None
# This will be assigned to the scrolledtext widget for logging
compliance_log = None
editor_log = None
section_synthesis_log = None
notebook = None # Make notebook global to access in show_main_app

compliance_tab_frame = None
word_editor_tab_frame = None
section_synthesis_tab_frame = None

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
word_editor_output_display = None

# A dictionary to hold references to tab frames for easier lookup
tab_frames = {}

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
    if word_editor_output_display: word_editor_output_display.config(state=tk.DISABLED)
    
    if editing_manual_path_label: editing_manual_path_label.config(state=tk.DISABLED)
    if section_path_label: section_path_label.config(state=tk.DISABLED)
    if start_compliance_button: start_compliance_button.config(state=tk.DISABLED)
    if guidelines_button: guidelines_button.config(state=tk.DISABLED)
    if manual_compliance_button: manual_compliance_button.config(state=tk.DISABLED)

def initialize_llm_and_embeddings(temp):
    global llm, embeddings, compliance_log, editor_log
    try:
        if llm is None:
            log(f"Initializing LLM with LM Studio from: {LM_STUDIO_BASE_URL}\n")
            llm = ChatOpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio", model="local-model", temperature=temp)
        if embeddings is None:
            log("Initializing Embeddings with CPU-Based HuggingFaceEmbeddings.\n")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        return True
    except Exception as e:
        log(f"Error initializing LLM/Embeddings: {e}\n")
        messagebox.showerror("Initialization Error", f"Failed to initialize LLM or Embeddings. Ensure LM Studio is running and accessible.\nError: {e}")
        return False

def select_editing_manual_file():
    global selected_editing_manual_path
    filepath = filedialog.askopenfilename(
        title="Select Manual Document to Edit",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_editing_manual_path = filepath
        editing_manual_path_label.config(text=f"File: {os.path.basename(filepath)}")
        log(f"Selected Manual for Editing: {filepath}\n")

def run_word_editor_threaded():
    disable_all_buttons()  # Disable buttons to prevent multiple clicks
    if not selected_editing_manual_path:
        messagebox.showwarning("Missing Input", "Please select a Document to edit.")
        if start_editing_button: start_editing_button.config(state=tk.NORMAL)
        if editing_manual_path_label: editing_manual_path_label.config(state=tk.NORMAL)
        return

    if not initialize_llm_and_embeddings(EDITING_TEMPERATURE):
        if start_editing_button: start_editing_button.config(state=tk.NORMAL)
        if editing_manual_path_label: editing_manual_path_label.config(state=tk.NORMAL)
        return

    if editor_log: editor_log.delete(1.0, tk.END) # Clear previous logs in the main log area

    import threading
    editing_thread = threading.Thread(target=perform_word_editing_logic)
    editing_thread.start()

def perform_word_editing_logic():
    global llm, editor_log, selected_editing_manual_path, word_editor_output_display
    try:
        log("Starting document editing process...\n")

        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            log(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.\n")
        else:
            required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
            for font_file in required_fonts:
                if not os.path.exists(os.path.join(font_dir, font_file)):
                    log(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.\n")

        compiled_reports_list = []

        full_filepath_to_edit = selected_editing_manual_path

        if not os.path.exists(full_filepath_to_edit):
            messagebox.showerror("File Not Found", f"Error: file to edit '{os.path.basename(full_filepath_to_edit)}' not found. Please ensure it exists. Exiting.")
            log(f"Error: file to edit '{os.path.basename(full_filepath_to_edit)}' not found. Exiting.\n")
            return

        raw_document_pages, original_document_name = load_document_content(full_filepath_to_edit)

        if not raw_document_pages:
            messagebox.showerror("Loading Error", f"Error: No content loaded from '{os.path.basename(full_filepath_to_edit)}'.")
            log( f"Error: No content loaded from '{os.path.basename(full_filepath_to_edit)}'. Aborting.\n")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=WORD_EDITOR_CHUNK_SIZE,
            length_function=len,
        )

        document_chunks = text_splitter.split_documents(raw_document_pages)
        log(f"Document split into {len(document_chunks)} chunks.\n")

        output_directory_base = "./AI_Tool_Editing_Results"
        os.makedirs(output_directory_base, exist_ok=True)

        individual_reports_dir = os.path.join(output_directory_base, f"{os.path.splitext(original_document_name)[0]}_edited_chunks")
        os.makedirs(individual_reports_dir, exist_ok=True)
        log(f"Individual edited chunks will be saved in: {individual_reports_dir}\n")

        full_edited_content_for_display = []

        for i, chunk_doc in enumerate(document_chunks):
            chunk_identifier = i + 1
            log(f"\n--- Starting Analysis for Chunk {chunk_identifier} ({len(chunk_doc.page_content)} characters) ---\n")

            analysis_result = editor_analyze_single_chunk(chunk_doc.page_content, chunk_identifier, llm)
            compiled_reports_list.append(analysis_result)

            individual_pdf_output_filename = os.path.join(individual_reports_dir, f"edited_chunk_{chunk_identifier}.pdf")

            write_edited_individual_pdf(
                analysis_result["llm_response"],
                analysis_result["chunk_id"],
                original_document_name,
                individual_pdf_output_filename
            )
            log(f"Saved individual edited content for '{analysis_result['chunk_id']}' to: {individual_pdf_output_filename}\n")
            log(f"\n{'='*20} End of Analysis for {analysis_result['chunk_id']} {'='*20}\n")

            # Append edited content for display in the GUI
            # analysis_result["llm_response"] is already a string here
            full_edited_content_for_display.append(f"\n--- {analysis_result['chunk_id']} ---\n{analysis_result['llm_response'].strip()}\n") # MODIFIED LINE: Removed .content and added .strip()

        # Update the word editor output display on the main thread
        root.after(0, word_editor_output_display.delete, 1.0, tk.END)
        root.after(0, word_editor_output_display.insert, tk.END, "\n".join(full_edited_content_for_display))
        root.after(0, word_editor_output_display.see, tk.END)

        log("\n--- Generating Consolidated Edited Report ---\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_edited_{os.path.splitext(original_document_name)[0]}_{timestamp}.pdf")
        try:
            write_edited_consolidated_pdf(compiled_reports_list, original_document_name, consolidated_pdf_filename)
            log(f"Successfully saved consolidated edited report to PDF: {consolidated_pdf_filename}\n")
            messagebox.showinfo("Process Complete", f"Document editing finished!\nReports saved to:\n{os.path.abspath(output_directory_base)}")
        except Exception as e:
            log(f"Error saving consolidated PDF report: {e}\n")
            messagebox.showerror("Error Saving Report", f"Could not save consolidated report.\nError: {e}")

    except Exception as e:
        log(f"An unexpected error occurred during document editing: {e}\n")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")
    finally:
        if start_editing_button: start_editing_button.config(state=tk.NORMAL)
        log("\nWord Editor Program Finished.\n")
    
def select_guidelines_file():
    global selected_guidelines_path
    filepath = filedialog.askopenfilename(
        title="Select Guidelines Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_guidelines_path = filepath
        guidelines_path_label.config(text=f"Guidelines: {os.path.basename(filepath)}")
        log( f"Selected Guidelines: {filepath}\n")

def select_manual_files():
    global selected_compliance_manual_path
    filepath = filedialog.askopenfilename(
        title="Select Manual Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_compliance_manual_path=filepath
        compliance_manual_label.config(text=f"Manual: {os.path.basename(filepath)}")
        log(f"Selected Manual File: {filepath}\n")

def run_compliance_check_threaded():
    disable_all_buttons()  # Disable all buttons during compliance check

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

    if not initialize_llm_and_embeddings(COMPLIANCE_TEMPERATURE):
        if start_compliance_button: start_compliance_button.config(state=tk.NORMAL)
        if guidelines_button: guidelines_button.config(state=tk.NORMAL)
        if manual_compliance_button: manual_compliance_button.config(state=tk.NORMAL)
        return

    import threading
    compliance_thread = threading.Thread(target=perform_compliance_logic, args=(compliance_aspects_from_user,))
    compliance_thread.start()

def perform_compliance_logic(user_defined_compliance_aspects):
    global llm, embeddings, compliance_log
    try:
        log("Starting compliance check process...\n")

        # FONT SETUP FOR THE OUTPUT PDF REPORTS

        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            log(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.\n")
        else:
            required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
            for font_file in required_fonts:
                if not os.path.exists(os.path.join(font_dir, font_file)):
                    log(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.\n")

        # --- Load and Process Guidelines ---
        log(f"Loading guidelines from: {selected_guidelines_path}\n")
        raw_guidelines_docs, _ = load_document_content(selected_guidelines_path)
        if not raw_guidelines_docs:
            log(f"Error: No content loaded from guidelines document '{os.path.basename(selected_guidelines_path)}'. Aborting.\n")
            return

        # Using a fixed chunk size for guidelines, if different from COMPLIANCE_CHECKER_CHUNK_SIZE
        guidelines_chunks = split_documents_into_chunks(raw_guidelines_docs, chunk_size=700, chunk_overlap=150)
        log("Creating Vector Store For Guidelines Document...\n")
        guidelines_vectorstore = create_vector_store(guidelines_chunks, embeddings)
        log(f"Guidelines document '{os.path.basename(selected_guidelines_path)}' processed and indexed into vector store.\n")

        compiled_reports_list = []
        output_directory_base = "./AI_Tool_Compliance_Results"
        os.makedirs(output_directory_base, exist_ok=True)
        individual_reports_dir = os.path.join(output_directory_base, "individual_compliance_reports")
        os.makedirs(individual_reports_dir, exist_ok=True)
        log(f"Individual compliance reports will be saved in: {individual_reports_dir}\n")

        # --- PROCESS THE SINGLE MANUAL DOCUMENT ---
        if not selected_compliance_manual_path:
            log("Error: No manual document selected for compliance check. Aborting.\n")
            return

        # Assuming selected_compliance_manual_paths contains only one path, or we take the first one
        full_filepath = selected_compliance_manual_path # Take the first selected file
        manual_filename_short = os.path.basename(full_filepath)
        log(f"\n--- Processing Manual Document: {manual_filename_short} ---\n")

        # Load the entire manual document
        raw_manual_docs, loaded_manual_filename = load_document_content(full_filepath)

        if not raw_manual_docs:
            log(f"Skipping processing for {manual_filename_short} due to loading error or empty content.\n")
            return

        # Join all pages/sections of the manual into one long string to be split by RecursiveCharacterTextSplitter
        full_manual_text = "\n\n".join([doc.page_content for doc in raw_manual_docs])
        log(f"Full manual '{loaded_manual_filename}' loaded with {len(full_manual_text)} characters.\n")

        # Initialize RecursiveCharacterTextSplitter with global chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=COMPLIANCE_CHECKER_CHUNK_SIZE,
            chunk_overlap=int(COMPLIANCE_CHECKER_CHUNK_SIZE * 0.2) # Typically 20% overlap
        )
        # Create chunks from the full manual text
        manual_text_chunks = text_splitter.create_documents([full_manual_text])
        log(f"Manual document split into {len(manual_text_chunks)} chunks using RecursiveCharacterTextSplitter (chunk size: {COMPLIANCE_CHECKER_CHUNK_SIZE}).\n")

        # --- Process Each Manual Chunk ---
        for i, chunk_doc in enumerate(manual_text_chunks):
            chunk_display_name = f"{loaded_manual_filename}_chunk_{i+1}"
            log(f"\n--- Analyzing Manual Chunk {i+1}/{len(manual_text_chunks)} ({chunk_display_name}) ---\n")
            manual_chunk_full_text = chunk_doc.page_content # The content of the current chunk

            log(f"LLM: Identifying relevant aspects for {chunk_display_name}...\n")
            relevant_aspects_for_this_chunk = get_llm_relevant_aspects(
                manual_chunk_full_text,
                user_defined_compliance_aspects,
                llm
            )

            if not relevant_aspects_for_this_chunk:
                log(f"LLM found no relevant compliance aspects for '{chunk_display_name}'. Skipping compliance check for this chunk.\n")
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

            log(f"LLM identified relevant aspects for '{chunk_display_name}':\n")
            for aspect in relevant_aspects_for_this_chunk:
                log(f" - {aspect}\n")

            log(f"Starting Compliance Analysis for {chunk_display_name}...\n")

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
            log(f"Saved individual compliance report for '{chunk_display_name}' to: {individual_pdf_output_filename}\n")
            log(f"\n{'='*20} End of Analysis for '{chunk_display_name}' {'='*20}\n")

        # --- Consolidated Report Generation (remains similar) ---
        log("\n--- Generating Consolidated Compliance Report ---\n")
        consolidated_pdf_filename = os.path.join(output_directory_base, f"consolidated_compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        try:
            write_report_to_pdf(compiled_reports_list, consolidated_pdf_filename)
            log(f"Successfully saved consolidated compliance report to PDF: {consolidated_pdf_filename}\n")
            messagebox.showinfo("Process Complete", f"Compliance check finished!\nReports saved to:\n{os.path.abspath(output_directory_base)}")
        except Exception as e:
            log(f"Error saving consolidated PDF report: {e}\n")
            messagebox.showerror("Error Saving Report", f"Could not save consolidated report.\nError: {e}")

    except Exception as e:
        log(f"An unexpected error occurred during the compliance check: {e}\n")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")
    finally:
        # Check if the buttons exist before trying to configure them
        if 'start_compliance_button' in globals() and start_compliance_button.winfo_exists():
            start_compliance_button.config(state=tk.NORMAL)
        if 'guidelines_button' in globals() and guidelines_button.winfo_exists():
            guidelines_button.config(state=tk.NORMAL)
        if 'manual_button' in globals() and manual_compliance_button.winfo_exists(): # Assuming this is your add manual file button
            manual_compliance_button.config(state=tk.NORMAL)
        log("\nProgram Finished.\n")

def select_section_file():
    global selected_section_path, section_path_label, section_synthesis_log # Ensure section_synthesis_log is global
    filepath = filedialog.askopenfilename(
        title="Select Source Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_section_path = filepath
        # Ensure the label widget exists before trying to update it
        if section_path_label:
            section_path_label.config(text=f"Source: {os.path.basename(filepath)}")
        log(f"Selected Source Document for Synthesis: {filepath}\n")

def select_sample_file():
    global selected_sample_path, sample_path_label, section_synthesis_log # Ensure section_synthesis_log is global
    filepath = filedialog.askopenfilename(
        title="Select Sample Text Document",
        filetypes=[("Document Files", "*.pdf *.docx *.txt")]
    )
    if filepath:
        selected_sample_path = filepath
        # Ensure the label widget exists before trying to update it
        if sample_path_label:
            sample_path_label.config(text=f"Sample: {os.path.basename(filepath)}")
        log(f"Selected Sample Text Document: {filepath}\n")


def perform_section_synthesis_logic():
    global llm, section_synthesis_log, selected_section_path, selected_sample_path # Ensure global access

    try:
        log(f"Starting section synthesis process...\n")

        # --- Font Directory Check (for PDF generation) ---
        font_dir = "./fonts"
        if not os.path.exists(font_dir):
            log(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.\n")
        else:
            required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
            for font_file in required_fonts:
                if not os.path.exists(os.path.join(font_dir, font_file)):
                    log(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.\n")

        compiled_reports_list = []
        output_directory_base = "./AI_Tool_Section_Synthesis_Results"
        os.makedirs(output_directory_base, exist_ok=True)
        log(f"Output directory created: {output_directory_base}\n")

        # --- Load Source Section Document ---
        if not selected_section_path:
            log(f"Error: No source document selected for synthesis. Aborting.\n")
            messagebox.showwarning("Missing Input", "Please select a Source Document for synthesis.")
            return

        log(f"Loading source section from: {selected_section_path}\n")
        raw_section_docs, section_filename_short = load_document_content(selected_section_path)
        if not raw_section_docs:
            log(f"Error: No content loaded from source document '{section_filename_short}'. Aborting.\n")
            messagebox.showerror("Loading Error", f"Could not load content from source document: {section_filename_short}")
            return

        full_section_text = "\n\n".join([doc.page_content for doc in raw_section_docs])
        log(f"Full source section '{section_filename_short}' loaded with {len(full_section_text)} characters.\n")

        # --- Load Sample Text Document ---
        if not selected_sample_path:
            log(f"Error: No sample document selected for style match. Aborting.\n")
            messagebox.showwarning("Missing Input", "Please select a Sample Text Document for style match.")
            return

        log(f"Loading sample text from: {selected_sample_path}\n")
        raw_sample_docs, sample_filename_short = load_document_content(selected_sample_path)
        if not raw_sample_docs:
            log(f"Error: No content loaded from sample document '{sample_filename_short}'. Aborting.\n")
            messagebox.showerror("Loading Error", f"Could not load content from sample document: {sample_filename_short}")
            return

        full_sample_text = "\n\n".join([doc.page_content for doc in raw_sample_docs])
        log(f"Full sample text '{sample_filename_short}' loaded with {len(full_sample_text)} characters.\n")

        # --- Perform the Synthesis ---
        log(f"Calling LLM for section synthesis...\n")
        analysis_result = section_synthesis(full_section_text, full_sample_text, llm)
        compiled_reports_list.append({"section_filename": section_filename_short, "synthesis_result": analysis_result}) # Store as dict for clearer report

        log("\n--- Generating Consolidated Synthesis Report ---\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Use the name of the section file for the output PDF
        output_base_name = os.path.splitext(section_filename_short)[0]
        consolidated_pdf_filename = os.path.join(output_directory_base, f"synthesized_report_for_{output_base_name}_{timestamp}.pdf")

        try:
            # Assuming write_edited_consolidated_pdf handles a list of dicts now
            # You might need to adjust write_edited_consolidated_pdf to accept the new compiled_reports_list format
            write_synthesis_report_to_pdf(analysis_result, section_filename_short, consolidated_pdf_filename)
            log(f"Successfully saved synthesized section to PDF: {consolidated_pdf_filename}\n")
            messagebox.showinfo("Process Complete", f"Section Synthesis finished!\nOutput saved to:\n{os.path.abspath(output_directory_base)}")
        except Exception as e:
            log(f"Error saving consolidated PDF report: {e}\n")
            messagebox.showerror("Error Saving Report", f"Could not save consolidated report.\nError: {e}")

    except Exception as e:
        log(f"An unexpected error occurred during section synthesis: {e}\n")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")
    finally:
        # Re-enable buttons specific to this tab
        if start_synthesis_button and start_synthesis_button.winfo_exists(): start_synthesis_button.config(state=tk.NORMAL)
        if section_button and section_button.winfo_exists(): section_button.config(state=tk.NORMAL)
        if sample_button and sample_button.winfo_exists(): sample_button.config(state=tk.NORMAL)
        log("\nSection Synthesis Program Finished.\n")


def run_synthesis_threaded():
    disable_all_buttons() # Disable all relevant buttons across the app

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

    if not initialize_llm_and_embeddings(EDITING_TEMPERATURE):
        if start_synthesis_button: start_synthesis_button.config(state=tk.NORMAL)
        if section_button: section_button.config(state=tk.NORMAL)
        if sample_button: sample_button.config(state=tk.NORMAL)
        return

    # Clear the log for this tab specifically
    if section_synthesis_log: section_synthesis_log.delete(1.0, tk.END)

    import threading
    synthesis_thread = threading.Thread(target=perform_section_synthesis_logic)
    synthesis_thread.start()

# --- GUI Setup ---

# Functions to initialize the main app components (notebook and tabs)

def create_compliance_checker_tab_frame(parent):
    compliance_tab_frame = ttk.Frame(parent, padding="10")

    file_frame = ttk.LabelFrame(compliance_tab_frame, text="Document Selection", padding="10")
    file_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    global guidelines_path_label, guidelines_button # Declare globals here, define later
    ttk.Label(file_frame, text="Guidelines Document:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    guidelines_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    guidelines_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    guidelines_button = ttk.Button(file_frame, text="Browse Files", command=select_guidelines_file)
    guidelines_button.grid(row=0, column=2, padx=5, pady=5)

    global compliance_manual_label, manual_compliance_button # Declare globals here
    ttk.Label(file_frame, text="Manual Documents:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    compliance_manual_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    compliance_manual_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    manual_compliance_button = ttk.Button(file_frame, text="Browse Files", command=select_manual_files)
    manual_compliance_button.grid(row=1, column=2, padx=5, pady=5, sticky="nw")

    global compliance_aspects_input # Declare global here
    ttk.Label(file_frame, text="Compliance Aspects (one per line):").grid(row=2, column=0, padx=5, pady=5, sticky="nw")
    compliance_aspects_input = scrolledtext.ScrolledText(file_frame, height=5, width=50, wrap=tk.WORD)
    compliance_aspects_input.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
    compliance_aspects_input.insert(tk.END, """Functions, members, and procedures of the Social Security Board\nResponsibilities of the Administrator\nAudit Requirements and External Oversight\nEmployee contribution amounts and regulations\nClassification of self-employed peoples\nClassification of disabilities\nEmployer contribution amounts and regulations\nThe keeping of accounts and reports\nAppeals and Dispute Resolution Mechanisms\nEmployee offenses and penalties\nHealth insurance benefits\nExclusions and limitations of health insurance coverage\nReimbursement procedures and requirements\nSubscription and enrollment processes\nPrivacy and data protection policies\nSuccession and transfer of medical savings account after death\nDue dates""")

    file_frame.columnconfigure(1, weight=1)

    control_frame = ttk.LabelFrame(compliance_tab_frame, text="Controls and Output", padding="10")
    control_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def start_compliance_check_button_action():
        log(f"Starting compliance check with {len(selected_compliance_manual_path)} manual documents and guidelines from {selected_guidelines_path}.\n")
        run_compliance_check_threaded()

    global start_compliance_button 
    start_compliance_button = ttk.Button(control_frame, text="Start Compliance Check", command=start_compliance_check_button_action)
    start_compliance_button.pack(pady=10)

    # Create a LabelFrame specifically for the log
    log_frame = ttk.LabelFrame(control_frame, text="Program Information Log", padding="5")
    log_frame.pack(expand=True, fill="both", padx=5, pady=5)

    global compliance_log 
    compliance_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    compliance_log.pack(expand=True, fill="both")
    compliance_log.see(tk.END)

    compliance_tab_frame.grid_columnconfigure(0, weight=1)
    compliance_tab_frame.grid_columnconfigure(1, weight=1)
    compliance_tab_frame.grid_rowconfigure(1, weight=1)

    return compliance_tab_frame

def create_word_editor_tab_frame(parent):
    # This frame will hold all Word Editor UI
    word_editor_tab_frame = ttk.Frame(parent, padding="10")

    # Frame for file selection
    file_frame = ttk.LabelFrame(word_editor_tab_frame, text="Select Document for Editing", padding="10")
    file_frame.pack(pady=10, padx=10, fill="x")

    ttk.Label(file_frame, text="Document to Edit:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    global editing_manual_path_label # Global for updating text
    editing_manual_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    editing_manual_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    ttk.Button(file_frame, text="Browse File", command=select_editing_manual_file).grid(row=0, column=2, padx=5, pady=5)
    file_frame.columnconfigure(1, weight=1)

    # Control frame
    control_frame = ttk.LabelFrame(word_editor_tab_frame, text="Controls", padding="10")
    control_frame.pack(pady=10, padx=10, fill="x")

    def start_editing_button_action():
        log(f"Starting AI editing for document: {selected_editing_manual_path}\n")
        run_word_editor_threaded()

    global start_editing_button # Global for enabling/disabling
    start_editing_button = ttk.Button(control_frame, text="Start AI Editing", command=start_editing_button_action)
    start_editing_button.pack(pady=10)

    # Output display area
    output_frame = ttk.LabelFrame(word_editor_tab_frame, text="Edited Content Output", padding="10")
    output_frame.pack(pady=10, padx=10, fill="both", expand=True)

    # Create a LabelFrame specifically for the log
    log_frame = ttk.LabelFrame(control_frame, text="Program Information Log", padding="5")
    log_frame.pack(expand=True, fill="both", padx=5, pady=5) # Pack the new log_frame into control_frame

    global editor_log # Set global gui_log here
    editor_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD) # Parent is now log_frame
    editor_log.pack(expand=True, fill="both") # No need for padx/pady here, as log_frame has padding
    editor_log.see(tk.END) # Scroll to the end to show the initial message

    word_editor_tab_frame.grid_columnconfigure(0, weight=1)
    word_editor_tab_frame.grid_columnconfigure(1, weight=1)
    word_editor_tab_frame.grid_rowconfigure(1, weight=1)

    global word_editor_output_display # Global to write edited text to
    word_editor_output_display = scrolledtext.ScrolledText(output_frame, height=20, wrap=tk.WORD)
    word_editor_output_display.pack(expand=True, fill="both")

    return word_editor_tab_frame

def create_section_synthesis_tab_frame(parent):
    global section_synthesis_log, section_path_label, section_button, sample_path_label, sample_button, start_synthesis_button

    section_synthesis_tab_frame = ttk.Frame(parent, padding="10")

    # --- Document Selection Frame ---
    file_frame = ttk.LabelFrame(section_synthesis_tab_frame, text="Document Selection", padding="10")
    file_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    # Source Document Selection
    ttk.Label(file_frame, text="Source Document:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    section_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    section_path_label.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    # FIX: Changed command to select_section_file
    section_button = ttk.Button(file_frame, text="Browse Source", command=select_section_file)
    section_button.grid(row=0, column=2, padx=5, pady=5)

    # Sample Text Document Selection
    ttk.Label(file_frame, text="Sample Text Document:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    sample_path_label = ttk.Label(file_frame, text="No file selected", wraplength=400)
    sample_path_label.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    # FIX: Changed command to select_sample_file
    sample_button = ttk.Button(file_frame, text="Browse Sample", command=select_sample_file)
    sample_button.grid(row=1, column=2, padx=5, pady=5)
    file_frame.columnconfigure(1, weight=1)

    # --- Controls Frame ---
    control_frame = ttk.LabelFrame(section_synthesis_tab_frame, text="Controls", padding="10")
    control_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def start_synthesis_button_action():
        log(f"Starting section synthesis with source document: {selected_section_path} and sample text: {selected_sample_path}\n")
        run_synthesis_threaded()

    start_synthesis_button = ttk.Button(control_frame, text="Start Synthesis", command=start_synthesis_button_action)
    start_synthesis_button.pack(pady=10)

    # --- Program Information Log for this tab ---
    # FIX: Changed parent from control_frame to section_synthesis_tab_frame
    log_frame = ttk.LabelFrame(section_synthesis_tab_frame, text="Program Information Log", padding="5")
    # Place this log frame below the control_frame, adjusting the row number
    log_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew") # Make it sticky to expand

    section_synthesis_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
    section_synthesis_log.pack(expand=True, fill="both")
    section_synthesis_log.insert(tk.END, "Section Synthesis Log Initialized.\n")
    section_synthesis_log.see(tk.END)

    # --- Grid configuration for section_synthesis_tab_frame ---
    section_synthesis_tab_frame.grid_columnconfigure(0, weight=1)
    section_synthesis_tab_frame.grid_columnconfigure(1, weight=1)
    section_synthesis_tab_frame.grid_rowconfigure(0, weight=0) # File selection frame
    section_synthesis_tab_frame.grid_rowconfigure(1, weight=0) # Controls frame
    section_synthesis_tab_frame.grid_rowconfigure(2, weight=1) # Log frame (this one expands)

    return section_synthesis_tab_frame

def initialize_main_app_components():
    global notebook, compliance_tab_frame, word_editor_tab_frame, tab_frames

    if notebook is None:
        notebook = ttk.Notebook(main_app_frame)
        notebook.pack(expand=True, fill="both")
        # No root.update_idletasks() here, as per your preference.
        # If issues persist, consider re-adding it after notebook.pack() and after notebook.add() calls.

        # Create the frames and assign them to the global variables
        compliance_tab_frame = create_compliance_checker_tab_frame(notebook)
        word_editor_tab_frame = create_word_editor_tab_frame(notebook)
        section_synthesis_tab_frame = create_section_synthesis_tab_frame(notebook)

        # Now, explicitly add these frames to the notebook with their text labels
        # And store them in the tab_frames dictionary
        notebook.add(compliance_tab_frame, text="Compliance Checker")
        tab_frames["Compliance Checker"] = compliance_tab_frame

        notebook.add(word_editor_tab_frame, text="Word Editor")
        tab_frames["Word Editor"] = word_editor_tab_frame

        notebook.add(section_synthesis_tab_frame, text="Section Synthesis")
        tab_frames["Section Synthesis"] = section_synthesis_tab_frame

def show_main_app(target_tab_key):
    home_frame.pack_forget()
    main_app_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Ensure the main app components (notebook and tabs) are initialized
    initialize_main_app_components()

    # Select the requested tab using the stored frame object
    if target_tab_key in tab_frames:
        notebook.select(tab_frames[target_tab_key])
    else:
        print(f"Warning: Attempted to select unknown tab: {target_tab_key}. Falling back to first tab.")
        notebook.select(0) # Fallback to first tab by index

def create_home_screen(parent):
    home_frame = ttk.Frame(parent, padding="20")
    home_frame.pack(fill="both", expand=True)

    ttk.Label(home_frame, text="Welcome to the AI Compliance and Editing Tool!",
              font=("Helvetica", 18, "bold")).pack(pady=20)

    ttk.Label(home_frame, text=(
        "Before proceeding, please do the following: please ensure you have LM Studio installed and open on your computer.\n\n"), justify="center", font=("Helvetica", 11)).pack(pady=10)
        
    ttk.Label(home_frame,text=("1.  In LM Studio, navigate to the 'Developer' tab on the left and select a local model from the dropdown at the top.\n"
        "   We suggest mistral-7b-instruct-v0.1 as it is a good balance of performance and speed.\n\n"
        "2.  Make sure the status is 'Running' and that the green text says 'READY'\n\n"
        "3.  LM Studio should have its local server 'Reachable at' http://localhost:1234/v1\n\n"
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