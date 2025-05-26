# MAJOR CHANGES FROM VERSION 5:
# * Identified section headers and stored them as metadata
# * Changed chunk retrieval to get 5 times the top_k amount (5), then filtered based on the section
#         if there were still too many chunks, top_k again.
# * Refine prompt

import os
import re # Import regex module
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./employee_manuals"

MANUAL_OLD_NAME = "Employee_Manual_2018.pdf"
MANUAL_NEWEST_NAME = "Employee_Manual_2023.pdf"

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "granite3-dense"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Define key policy areas/topics to analyze.
KEY_POLICY_AREAS = ["7. Employment"] # Example: targeting section 7

# --- Initialization ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.2)

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# --- Functions ---

# Helper function to add basic metadata (source, page/chunk)
# Section-specific metadata will be handled robustly in split_documents
def add_basic_metadata(docs, filename):
    for i, doc in enumerate(docs):
        doc.metadata["source"] = filename
        if 'page' in doc.metadata:
            doc.metadata["original_page"] = doc.metadata['page']
        else:
            doc.metadata["original_page"] = f"chunk_{i+1}"


def load_filename(documents, directory_path, filename):
    filepath = os.path.join(directory_path, filename)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        print(f"Skipping unsupported file type: {filename}")
        return []

    print(f"Loading {filename}...")
    docs = loader.load()
    add_basic_metadata(docs, filename) # Use basic metadata function
    documents.extend(docs)

def load_directory(documents, directory_path):
    for filename_in_dir in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename_in_dir)
            if filename_in_dir.endswith((".pdf", ".docx", ".txt")):
                loader = None
                if filename_in_dir.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filename_in_dir.endswith(".docx"):
                    loader = Docx2txtLoader(filepath)
                elif filename_in_dir.endswith(".txt"):
                    loader = TextLoader(filepath)

                if loader:
                    print(f"Loading {filename_in_dir}...")
                    docs = loader.load()
                    add_basic_metadata(docs, filename_in_dir) # Use basic metadata function
                    documents.extend(docs)
            else:
                print(f"Skipping unsupported file type: {filename_in_dir}")

def load_documents(directory_path, filename=None):
    documents = []
    if filename:
        load_filename(documents, directory_path, filename)
    else:
        load_directory(documents, directory_path)
    return documents

# --- split_documents (MODIFIED FOR FULL SECTION HEADING METADATA) ---
def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            r"\n\s*\d+\.\d+\.\d+\.\d+\.\d+\.\s", # Most specific (e.g., 7.1.1.1.1.)
            r"\n\s*\d+\.\d+\.\d+\.\d+\.\s",     # 7.1.1.1.
            r"\n\s*\d+\.\d+\.\d+\.\s",          # 7.1.1.
            r"\n\s*\d+\.\d+\.\s",               # 7.1.
            r"\n\s*\d+\.\s",                    # 7. (Main section e.g., "7. Employment")
            "\n\n",
            "\n",
            " ",
            "",
        ]
    )
    split_docs = text_splitter.split_documents(documents)

    # Dictionaries to keep track of the most recent section heading and number per source manual
    current_full_section_heading = {} # e.g., "7. Employment"
    current_section_number = {} # e.g., "7" or "7.1"

    current_full_section_heading = {} # e.g., "7. Employment"
    current_section_number_prefix = {} # e.g., "7" or "7.1"

    # Regex to find a heading: starts with optional whitespace, then a number (with optional sub-numbers),
    # followed by a period (optional), then whitespace, then the heading text.
    # Captures the number part in group 1 and the full heading line in group 0 (the whole match).
    # Use re.search for more flexibility within the beginning of the chunk
    # The `.*?` (non-greedy) and `$` (end of line) are important for single-line headings.
    # Add `\b` for word boundary to prevent matching numbers embedded in text.
    heading_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*([^\n]*)", re.MULTILINE)

    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")

        # Try to find a heading at the very start of the chunk's content
        # Only look at the first 200 characters to prevent matching later in a long chunk
        content_prefix = doc.page_content[:200]
        
        match = heading_pattern.search(content_prefix)

        if match:
            # Found a new heading!
            number_part = match.group(1).strip()
            full_heading_text = match.group(2).strip()
            
            # Construct the full heading, prioritizing the structure 'NUMBER. TEXT'
            # If the text part is empty or just whitespace, use the number itself
            full_heading = f"{number_part}. {full_heading_text}" if full_heading_text else number_part

            current_full_section_heading[source] = full_heading
            current_section_number_prefix[source] = number_part
            
        
        # Assign the most recently identified heading and number to the current chunk
        doc.metadata["full_section_heading"] = current_full_section_heading.get(source, "General Policy/Introduction")
        doc.metadata["section_number"] = current_section_number_prefix.get(source, "N/A") # This will be used for filtering

    return split_docs


def create_vector_store(chunks, embeddings_model):
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

# --- chunk_retrieval (MODIFIED FOR METADATA FILTERING & DISPLAY) ---
def chunk_retrieval(relevant_chunks_for_llm, manual_vector_stores_dict, policy_area, top_k):
    # Extract the main section number from the policy_area query for filtering
    # e.g., "7. Employment" -> "7"
    main_section_match = re.match(r"^(\d+(?:\.\d+)*)\.?", policy_area)
    target_section_prefix_for_filter = main_section_match.group(1) if main_section_match else None

    print(f"\n--- Retrieving Chunks for Policy Area: '{policy_area}' ---")
    print(f"Target section prefix for filtering: {target_section_prefix_for_filter}")

    for manual_name, vector_store in manual_vector_stores_dict.items():
        print(f"  Searching in {manual_name}...")
        # Step 1: Initial similarity search - retrieve more documents than final 'top_k'
        retrieved_docs_raw = vector_store.similarity_search(policy_area, k=top_k * 5) # Retrieve more for robust filtering
        print(f"    Initial similarity search returned {len(retrieved_docs_raw)} chunks.")

        # Step 2: Filter by 'section_number' metadata
        filtered_docs_by_section = []
        if target_section_prefix_for_filter and target_section_prefix_for_filter != "N/A":
            for doc in retrieved_docs_raw:
                doc_section_number = doc.metadata.get("section_number", "N/A")
                # Check if the chunk's section_number starts with the target prefix
                if doc_section_number.startswith(target_section_prefix_for_filter):
                    filtered_docs_by_section.append(doc)
        else:
            # If the policy_area query doesn't have a specific section number,
            # rely only on semantic similarity for the initial set.
            filtered_docs_by_section = retrieved_docs_raw
        
        print(f"    After section filtering, {len(filtered_docs_by_section)} chunks remain.")

        # Step 3: Take the top_k relevant documents after section filtering
        final_docs_for_llm = filtered_docs_by_section[:top_k]

        if final_docs_for_llm:
            print(f"  Final selection: {len(final_docs_for_llm)} chunks from {manual_name}")
            for doc in final_docs_for_llm:
                # Use the new 'full_section_heading' for context to the LLM
                relevant_chunks_for_llm.append(
                    f"--- Source: {manual_name}, Page/Chunk: {doc.metadata.get('original_page', 'N/A')}, Section: {doc.metadata.get('full_section_heading', doc.metadata.get('section_number', 'N/A'))} ---\n"
                    f"{doc.page_content}\n"
                )
        else:
            print(f"  No relevant chunks found in {manual_name} for '{policy_area}' after section filtering.")


# --- compile_and_compare_policy_area ---
def compile_and_compare_policy_area(policy_area, manual_vector_stores_dict, llm_model, top_k):
    print(f"\n--- Analyzing Policy Area: '{policy_area}' ---")
    relevant_chunks_for_llm = []

    chunk_retrieval(relevant_chunks_for_llm, manual_vector_stores_dict, policy_area, top_k)

    if not relevant_chunks_for_llm:
        return {"context_text": "", "llm_response": f"No information found for '{policy_area}' across any of the manuals."}

    context_text = "\n\n".join(relevant_chunks_for_llm)

    print(f"\n -- Context Text Provided to LLM for '{policy_area}':\n{context_text}\n--- End Context Text ---")

    prompt = f"""
    You are an expert HR policy analyst. Your task is to copy and paste the following excerpts from the manuals AND combine them. Review and compare **ONLY** the provided excerpts from ROPSSA's employee manuals concerning the policy area: "{policy_area}".

    **CRUCIAL CONSTRAINTS (STRICTLY ADHERE):**
    * **PART 1 (Synthesized Policy - Verbatim):**
        * Present all policy information related to "{policy_area}" from the provided excerpts.
        * **MUST** include **ALL** relevant sections and their exact headings (e.g., "7. Employment", "7.1. Hiring Process", "7.1.1. Policies") in **STRICT NUMERICAL ORDER**.
        * For each section, provide the **EXACT, VERBATIM** text from the source manuals.
        * If content is identical or nearly identical across manuals, **PRIORITIZE** and use the verbatim text from the **2023 manual**.
        * **DO NOT** rephrase, summarize, abbreviate, or add any new information.
        * **DO NOT** omit any information from the relevant sections.

    * **PART 2 (Contradictions & Changes):**
        * List **ALL** contradictions or significant changes identified **within the provided excerpts**. If none, state "No contradictions or significant changes identified."
        * For each change, use this **EXACT FORMAT**:
            **Contradiction/Change in [Specific Policy Aspect]:**
            * **[2018 Manual - Section No. & Heading]:** "[Exact Quote]"
            * **[2023 Manual - Section No. & Heading]:** "[Exact Quote]"
            * **Significance:** [Brief explanation **based ONLY on provided texts**.]

    ---
    **Policy Area:** {policy_area}

    **Excerpts from Employee Manuals:**
    ---
    {context_text}
    ---

    **Your Compiled Information and Analysis (Start with PART 1, then PART 2):**
    """

    try:
        response = llm_model.invoke(prompt)
        return {"context_text": context_text, "llm_response": response}
    except Exception as e:
        return {"context_text": context_text, "llm_response": f"Error processing '{policy_area}': {e}"}


# --- Helper Function to write report to a PDF ---
def write_report_to_pdf(report_data, output_filepath):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_dir = "./fonts"
    # Ensure these font files exist in your './fonts' directory
    regular_font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    italic_font_path = os.path.join(font_dir, "NotoSans-Italic.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    bolditalic_font_path = os.path.join(font_dir, "NotoSans-BoldItalic.ttf")

    # Check if font files exist before adding them, use fallbacks if not
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
    # Use NotoSans if added, else fallback
    try:
        pdf.set_font("NotoSans", "BI", 20)
    except:
        pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 12, "ROPSSA Employee Manuals - Compiled Policy Report", align='C')
    pdf.ln(15)

    try:
        pdf.set_font("NotoSans", "", 12)
    except:
        pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 8, "This report synthesizes information from various versions of ROPSSA's employee manuals and highlights any contradictions or significant changes identified by the LLM.")
    pdf.ln(10)

    for topic, data in report_data.items():
        llm_response_content = data["llm_response"]
        context_text_content = data["context_text"]

        if pdf.get_y() > (pdf.h - 60):
            pdf.add_page()
        try:
            pdf.set_font("NotoSans", "B", 16)
        except:
            pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, f"Policy Area: {topic}", 0, 'L')
        pdf.ln(4)

        try:
            pdf.set_font("NotoSans", "", 11)
        except:
            pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, llm_response_content.strip())
        pdf.ln(8)

        if context_text_content.strip():
            if pdf.get_y() > (pdf.h - 40):
                pdf.add_page()
            try:
                pdf.set_font("NotoSans", "B", 12)
            except:
                pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, "--- Raw Context Provided to LLM ---", 0, 'L')
            pdf.ln(2)
            try:
                pdf.set_font("NotoSans", "I", 9)
            except:
                pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 4, context_text_content.strip())
            pdf.ln(8)

        try:
            pdf.set_font("NotoSans", "I", 9)
        except:
            pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, "-" * 100, align='C')
        pdf.ln(8)


    pdf.output(output_filepath)


# --- Main Program Logic ---


if __name__ == "__main__":
    if not os.path.exists(DOCUMENT_LIBRARY_PATH):
        os.makedirs(DOCUMENT_LIBRARY_PATH)
        print(f"Created directory: {DOCUMENT_LIBRARY_PATH}")
    
    # Check for fonts
    font_dir = "./fonts"
    if not os.path.exists(font_dir):
        print(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.")
    else:
        # Check if font files exist
        required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
        for font_file in required_fonts:
            if not os.path.exists(os.path.join(font_dir, font_file)):
                print(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.")


    # Load and Process Manuals
    manual_O_docs = load_documents(DOCUMENT_LIBRARY_PATH, MANUAL_OLD_NAME)
    manual_N_docs = load_documents(DOCUMENT_LIBRARY_PATH, MANUAL_NEWEST_NAME)

    if not manual_O_docs or not manual_N_docs:
        print("Error: One or more employee manuals not found or empty. Please check paths and content. Exiting.")
        exit()

    # Split documents into chunks for vector store creation
    chunk_size = 800
    chunk_overlap = 150
    manual_O_chunks = split_documents(manual_O_docs, chunk_size, chunk_overlap)
    manual_N_chunks = split_documents(manual_N_docs, chunk_size, chunk_overlap)


    # Create individual vector stores for each manual for targeted retrieval
    manual_vector_stores = {
        MANUAL_OLD_NAME: create_vector_store(manual_O_chunks, embeddings),
        MANUAL_NEWEST_NAME: create_vector_store(manual_N_chunks, embeddings)
    }
    print("\nAll manuals loaded and individual vector stores created.")

    compiled_report_by_topic = {}
    print("\n--- Starting Compilation and Contradiction Detection ---")

    top_k = 5 # Retrieve top 5 relevant chunks from each manual for this policy area
    for topic in KEY_POLICY_AREAS:
        compiled_data = compile_and_compare_policy_area(topic, manual_vector_stores, llm, top_k)
        compiled_report_by_topic[topic] = compiled_data
        print(f"\n{'='*20} End of Analysis for '{topic}' {'='*20}\n")

    print("\n--- Full Compiled Report (Console Preview) ---")
    for topic, data in compiled_report_by_topic.items():
        print(f"\n### Policy Area: {topic}\n")
        print(data["llm_response"])
        print("\n--------------------------------------------------")

    output_directory = "./output_reports_version_6"
    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_filename = os.path.join(output_directory, f"report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_report_by_topic, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")