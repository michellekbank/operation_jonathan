# MAJOR CHANGES FROM VERSION 6:
# Add way more metadata including full heading, subheadings, and such
# Include this when feeding into the LLM
# Sort the data by number so it would be feeding the LLM the policies in numerical order


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
KEY_POLICY_AREAS = ["7. Employment",
   "8. Probationary Period and Status",
   "9. Job Positions",
   "10. Compensation Policies And Practices",
   "11. Pay Periods",
   "12. Pay Procedures",
   "13. Payroll Deductions",
   "14. Overtime Pay",
   "15. Holiday Work Compensation",
   "16. Leave Policies",
   "17. Retention of Leave",
   "18. Retirement",
   "19. Employment Benefits",
   "20. Performance Evaluation",
   "21. Separation",
   "22. Employee Discipline",
   "23. Disciplinary Procedures",
   "24. Nepotism"
]

# --- Initialization ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
# Set temperature to a low but non-zero value for a balance of detail and accuracy
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.1) # Adjusted temperature

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
            r"\n\s*\d+\.\d+\.\d+\.\d+\.\d+\.\s.*", # Match full line of most specific section
            r"\n\s*\d+\.\d+\.\d+\.\d+\.\s.*",     # Match full line of 7.1.1.1.
            r"\n\s*\d+\.\d+\.\d+\.\s.*",          # Match full line of 7.1.1.
            r"\n\s*\d+\.\d+\.\s.*",               # Match full line of 7.1.
            r"\n\s*\d+\.\s.*",                    # Match full line of 7. (Main section e.g., "7. Employment")
            "\n\n",
            "\n",
            " ",
            "",
        ]
    )
    split_docs = text_splitter.split_documents(documents)

    # Use a list to store processed chunks
    processed_chunks = []

    # Dictionaries to keep track of the most recent section heading and number prefix per source manual
    current_full_section_heading = {} # e.g., "7. Employment"
    current_section_number_prefix = {} # e.g., "7", "7.1"

    # Regex to find a heading: starts with optional whitespace, then a number (with optional sub-numbers),
    # followed by a period (optional), then whitespace, then the heading text.
    # This one tries to capture the entire line to ensure `full_heading` is robust.
    heading_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*([^\n]*)", re.MULTILINE)

    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")
        original_page = doc.metadata.get("original_page", "N/A")

        # Try to find a heading at the very start of the chunk's content
        # Look at the first 200 characters, but prioritize a match at the absolute beginning
        content_prefix = doc.page_content[:200]

        match = heading_pattern.search(content_prefix)

        if match:
            # Found a new heading!
            number_part = match.group(1).strip()
            text_part = match.group(2).strip()
            full_heading = f"{number_part}. {text_part}" if text_part else number_part

            current_full_section_heading[source] = full_heading
            current_section_number_prefix[source] = number_part
        
        # Assign the most recently identified heading and number to the current chunk
        # Ensure metadata keys are correctly set for *each* chunk
        doc.metadata["full_section_heading"] = current_full_section_heading.get(source, "General Policy/Introduction")
        doc.metadata["section_number"] = current_section_number_prefix.get(source, "N/A")
        
        processed_chunks.append(doc) # Add the processed chunk

    return processed_chunks


def create_vector_store(chunks, embeddings_model):
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

# --- chunk_retrieval (MODIFIED FOR METADATA FILTERING & DISPLAY) ---
def chunk_retrieval(relevant_chunks_by_manual_section, manual_vector_stores_dict, policy_area, top_k):
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
            # Store chunks in a structured way for pre-processing
            for doc in final_docs_for_llm:
                section_num = doc.metadata.get('section_number', 'N/A')
                full_section_head = doc.metadata.get('full_section_heading', 'N/A')
                page_info = doc.metadata.get('original_page', 'N/A')
                
                # Store by manual name, then section number, then page info
                if manual_name not in relevant_chunks_by_manual_section:
                    relevant_chunks_by_manual_section[manual_name] = {}
                
                if section_num not in relevant_chunks_by_manual_section[manual_name]:
                    relevant_chunks_by_manual_section[manual_name][section_num] = {
                        'full_heading': full_section_head,
                        'content': [],
                        'page_info': []
                    }
                relevant_chunks_by_manual_section[manual_name][section_num]['content'].append(doc.page_content)
                relevant_chunks_by_manual_section[manual_name][section_num]['page_info'].append(page_info)
        else:
            print(f"  No relevant chunks found in {manual_name} for '{policy_area}' after section filtering.")


# --- compile_and_compare_policy_area ---
def compile_and_compare_policy_area(policy_area, manual_vector_stores_dict, llm_model, top_k):
    print(f"\n--- Analyzing Policy Area: '{policy_area}' ---")
    
    # Store relevant chunks in a structured way:
    # { 'manual_name': { 'section_number': { 'full_heading': '...', 'content': [], 'page_info': [] } } }
    relevant_chunks_by_manual_section = {}

    chunk_retrieval(relevant_chunks_by_manual_section, manual_vector_stores_dict, policy_area, top_k)

    if not relevant_chunks_by_manual_section:
        return {"context_text": "", "llm_response": f"No information found for '{policy_area}' across any of the manuals."}

    # --- Pre-process context_text for the LLM ---
    # Sort section numbers for consistent presentation to the LLM
    all_section_numbers = set()
    for manual_data in relevant_chunks_by_manual_section.values():
        all_section_numbers.update(manual_data.keys())

    # Sort numerically (e.g., "7.1.10" comes after "7.1.9")
    # This custom sort key function handles dotted numbers correctly
    def sort_key(s):
        return [int(part) if part.isdigit() else part for part in s.split('.')]
    
    sorted_section_numbers = sorted(list(all_section_numbers), key=sort_key)

    context_text_parts = []
    
    # Iterate through sorted sections and manuals to build the context for the LLM
    for section_num in sorted_section_numbers:
        for manual_name in [MANUAL_OLD_NAME, MANUAL_NEWEST_NAME]: # Ensure consistent order of manuals
            if manual_name in relevant_chunks_by_manual_section and \
               section_num in relevant_chunks_by_manual_section[manual_name]:
                
                section_data = relevant_chunks_by_manual_section[manual_name][section_num]
                full_heading = section_data['full_heading']
                # Join content parts for the same section if it was split into multiple chunks
                content = "\n".join(section_data['content']).strip()
                page_info = ", ".join(map(str, sorted(list(set(section_data['page_info']))))) # Unique and sorted page info

                context_text_parts.append(
                    f"--- Source: {manual_name}, Page/Chunk: {page_info}, Section: {full_heading} ({section_num}) ---\n"
                    f"{content}\n"
                )
    
    context_text = "\n\n".join(context_text_parts)
    # --- End Pre-processing ---


    # print(f"\n -- Context Text Provided to LLM for '{policy_area}':\n{context_text}\n--- End Context Text ---")

    # Adjusted prompt for even more specific instructions
    prompt = f"""
    You are an expert HR policy analyst. Review and compare the **EXACT, VERBATIM** content from the provided excerpts of ROPSSA's employee manuals concerning the policy area: "{policy_area}".

    **YOUR OUTPUT MUST FOLLOW THIS STRUCTURE AND THESE CRUCIAL CONSTRAINTS (STRICTLY ADHERE):**

    **PART 1: Synthesized Policy - Verbatim**
    * Present **ALL** policy information related to "{policy_area}" from the provided excerpts.
    * **MUST** include **EVERY DISTINCT NUMBERED SECTION** (e.g., 7.1, 7.1.1, 7.1.1.1, 7.1.1.5.1, 7.1.1.5.2, etc.) that appears in the "Excerpts from Employee Manuals" section below.
    * List these sections in **STRICT NUMERICAL ORDER** (e.g., 7. then 7.1. then 7.1.1. then 7.1.1.1. then 7.1.1.2. then 7.1.1.3. then 7.1.1.3.1. then 7.1.1.5.1. then 7.1.1.5.2. etc.).
    * For each listed section, provide its **EXACT, VERBATIM** text from the source manuals.
    * If a section's content is identical or nearly identical across both manuals, **PRIORITIZE** and use the verbatim text from the **2023 manual**.
    * **DO NOT** rephrase, summarize, abbreviate, or add any new information.
    * **DO NOT** omit any information from the relevant sections.

    **PART 2: Contradictions & Significant Changes**
    * List **ALL** contradictions or significant changes identified **within the provided excerpts**.
    * If no contradictions or significant changes are identified, state: "No contradictions or significant changes identified."
    * For each change, use this **EXACT FORMAT**:
        **Contradiction/Change in [Specific Policy Aspect]:**
        * **[2018 Manual - Section No. & Heading]:** "[Exact Quote from 2018 Manual]"
        * **[2023 Manual - Section No. & Heading]:** "[Exact Quote from 2023 Manual]"
        * **Significance:** [Brief explanation **based ONLY on provided texts**, explaining the nature of the difference.]

    ---
    **Excerpts from Employee Manuals (FOR YOUR REFERENCE AND VERBATIM COPY-PASTE):**
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
    chunk_overlap = 100
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

    output_directory = "./output_reports_version_7"
    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_filename = os.path.join(output_directory, f"report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_report_by_topic, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")