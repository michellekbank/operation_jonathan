# If headings were inconsistent, use the ones from 2023 instead of 2018

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
# Set temperature to a low but non-zero value for a balance of detail and accuracy
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.1) # Keep this temperature

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# --- Functions ---

# Helper function to add basic metadata (source, page/chunk)
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
    heading_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*([^\n]*)", re.MULTILINE)

    for doc in split_docs:
        source = doc.metadata.get("source", "unknown")
        original_page = doc.metadata.get("original_page", "N/A")

        # Try to find a heading at the very start of the chunk's content
        content_prefix = doc.page_content[:200] # Look at the first 200 chars for a heading
        
        match = heading_pattern.search(content_prefix)

        if match:
            # Found a new heading!
            number_part = match.group(1).strip()
            text_part = match.group(2).strip()
            full_heading = f"{number_part}. {text_part}" if text_part else number_part

            current_full_section_heading[source] = full_heading
            current_section_number_prefix[source] = number_part
        
        # Assign the most recently identified heading and number to the current chunk
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
                
                # Use section_num as the primary key for content, full_heading as metadata
                # Concatenate content if multiple chunks belong to the same section_num
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
    
    relevant_chunks_by_manual_section = {}
    chunk_retrieval(relevant_chunks_by_manual_section, manual_vector_stores_dict, policy_area, top_k)

    if not relevant_chunks_by_manual_section:
        return {"context_text": "", "llm_response": f"No information found for '{policy_area}' across any of the manuals."}

    # --- Pre-process context_text for the LLM ---
    # 1. Gather all unique section numbers from both manuals
    all_section_numbers = set()
    for manual_data in relevant_chunks_by_manual_section.values():
        all_section_numbers.update(manual_data.keys())

    # 2. Sort section numbers numerically
    def sort_key(s):
        return [int(part) if part.isdigit() else part for part in s.split('.')]
    
    sorted_section_numbers = sorted(list(all_section_numbers), key=sort_key)

    part1_context_parts = [] # For the highly structured PART 1 context
    full_raw_context_parts = [] # For the general raw context for PART 2 comparison

    # 3. Populate part1_context_parts and full_raw_context_parts
    for section_num in sorted_section_numbers:
        content_2023 = None
        content_2018 = None
        heading_2023 = None
        heading_2018 = None
        page_2023 = "N/A"
        page_2018 = "N/A"

        # Try to get content from 2023 manual first
        if MANUAL_NEWEST_NAME in relevant_chunks_by_manual_section and \
           section_num in relevant_chunks_by_manual_section[MANUAL_NEWEST_NAME]:
            section_data_2023 = relevant_chunks_by_manual_section[MANUAL_NEWEST_NAME][section_num]
            content_2023 = "\n".join(section_data_2023['content']).strip()
            heading_2023 = section_data_2023['full_heading']
            page_2023 = ", ".join(map(str, sorted(list(set(section_data_2023['page_info'])))))

        # Then from 2018 manual
        if MANUAL_OLD_NAME in relevant_chunks_by_manual_section and \
           section_num in relevant_chunks_by_manual_section[MANUAL_OLD_NAME]:
            section_data_2018 = relevant_chunks_by_manual_section[MANUAL_OLD_NAME][section_num]
            content_2018 = "\n".join(section_data_2018['content']).strip()
            heading_2018 = section_data_2018['full_heading']
            page_2018 = ", ".join(map(str, sorted(list(set(section_data_2018['page_info'])))))
        
        # Determine which content to use for PART 1 (prioritize 2023)
        content_for_part1 = content_2023 if content_2023 else content_2018
        heading_for_part1 = heading_2023 if heading_2023 else heading_2018
        
        if content_for_part1:
            part1_context_parts.append(f"{heading_for_part1}:\n{content_for_part1}")

        # Add to full_raw_context_parts for PART 2 reference (include both versions if available)
        if content_2018:
            full_raw_context_parts.append(
                f"--- Source: {MANUAL_OLD_NAME}, Page/Chunk: {page_2018}, Section: {heading_2018} ({section_num}) ---\n"
                f"{content_2018}\n"
            )
        if content_2023 and (content_2023 != content_2018 or heading_2023 != heading_2018): # Avoid duplicate if identical
             full_raw_context_parts.append(
                f"--- Source: {MANUAL_NEWEST_NAME}, Page/Chunk: {page_2023}, Section: {heading_2023} ({section_num}) ---\n"
                f"{content_2023}\n"
            )
    
    part1_context_str = "\n\n".join(part1_context_parts)
    full_raw_context_str = "\n\n".join(full_raw_context_parts)
    # --- End Pre-processing ---


    print(f"\n -- Context Text Provided to LLM for '{policy_area}':\n{full_raw_context_str}\n--- End Context Text ---")

    # Adjusted prompt: Now, PART 1 directly receives the pre-formatted text.
    # PART 2 still refers to the `full_raw_context_str`.
    prompt = f"""
    You are an expert HR policy analyst. Your task is to accurately present policy information and identify contradictions.

    **YOUR OUTPUT MUST FOLLOW THIS STRUCTURE AND THESE CRUCIAL CONSTRAINTS (STRICTLY ADHERE):**

    **PART 1: Synthesized Policy - Verbatim**
    * **REPRODUCE THE FOLLOWING TEXT EXACTLY AS PROVIDED. DO NOT REPHRASE, SUMMARIZE, OR OMIT ANYTHING.**
    * This text is already sorted and prioritized according to your requirements.

    {part1_context_str}

    **PART 2: Contradictions & Significant Changes**
    * Analyze the "Raw Excerpts" provided below to identify **ALL** contradictions or significant changes between the 2018 and 2023 manuals.
    * If no contradictions or significant changes are identified, state: "No contradictions or significant changes identified."
    * For each change, use this **EXACT FORMAT**:
        **Contradiction/Change in [Specific Policy Aspect]:**
        * **[2018 Manual - Section No. & Heading]:** "[Exact Quote from 2018 Manual]"
        * **[2023 Manual - Section No. & Heading]:** "[Exact Quote from 2023 Manual]"
        * **Significance:** [Brief explanation **based ONLY on provided texts**, explaining the nature of the difference.]

    ---
    **Raw Excerpts (FOR YOUR REFERENCE AND VERBATIM COPY-PASTE FOR PART 2):**
    ---
    {full_raw_context_str}
    ---

    **Your Compiled Information and Analysis (Start with PART 1, then PART 2):**
    """

    try:
        response = llm_model.invoke(prompt)
        return {"context_text": full_raw_context_str, "llm_response": response}
    except Exception as e:
        return {"context_text": full_raw_context_str, "llm_response": f"Error processing '{policy_area}': {e}"}

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

    top_k = 5 # Retrieve top 7 relevant chunks from each manual for this policy area
    for topic in KEY_POLICY_AREAS:
        compiled_data = compile_and_compare_policy_area(topic, manual_vector_stores, llm, top_k)
        compiled_report_by_topic[topic] = compiled_data
        print(f"\n{'='*20} End of Analysis for '{topic}' {'='*20}\n")

    print("\n--- Full Compiled Report (Console Preview) ---")
    for topic, data in compiled_report_by_topic.items():
        print(f"\n### Policy Area: {topic}\n")
        print(data["llm_response"])
        print("\n--------------------------------------------------")

    output_directory = "./output_reports_version_8"
    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_filename = os.path.join(output_directory, f"report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_report_by_topic, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")