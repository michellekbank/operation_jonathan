# MAJOR CHANGES FROM VERSION 8:
# Split documents actually based on section!!!!!
# First, identifies and groups based on distinct numbered sessions
# Then uses recursive character text splitter to make sure they are small enough MAKING SURE sub chunks keep the metadata
# Experimented with larger and smaller chunks… seemed to work better, so checked out chunk size 300, overlap 80

# -*-*-*-*- THIS IS THE BEST VERSION WE HAVE OF THE EMPLOYEE MANUAL PROCESSOR -*-*-*-*-

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
DOCUMENT_LIBRARY_PATH = "./employee_manuals"

MANUAL_OLD_NAME = "Employee_Manual_2018.pdf"
MANUAL_NEWEST_NAME = "Employee_Manual_2023.pdf"

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "granite3-dense"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Define key policy areas/topics to analyze.
KEY_POLICY_AREAS = [
    "Introduction",
    "1. Purpose and Scope",
    "2. Coverage",
    "3. Guiding Principles",
    "4. Governance",
    "5. Absence of Personnel Policies",
    "6. Position and Classes of Positions",
    "7. Employment",
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
    "24. Nepotism",
    "25. Social Security Employee Code of Conduct",
    "26. Revisions or Amendments"
]

# --- Initialization ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.1)

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# --- Functions ---

#     Loads content from a single document, returning raw pages with their numbers.
#     Each element in the list is a (page_content_string, page_number) tuple.
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
        return None, []

    print(f"Loading raw content from {os.path.basename(filepath)}...")
    raw_docs = loader.load()
    
    # Store page content and its original page number
    page_contents_with_metadata = []
    for doc in raw_docs:
        page_contents_with_metadata.append((doc.page_content, doc.metadata.get('page', 'N/A')))
    
    return page_contents_with_metadata, os.path.basename(filepath)

# --- REVISED split_documents for definitive section extraction ---
#     First, identifies and groups content by distinct numbered sections across pages.
#     Then, applies RecursiveCharacterTextSplitter to these logically grouped sections,
#     ensuring all sub-chunks inherit correct section metadata.
def split_documents(page_contents_with_metadata, filename, chunk_size=300, chunk_overlap=80):

    # Regex to find a heading: starts with optional whitespace, then a number (with optional sub-numbers),
    # followed by an optional period, then whitespace, then the heading text.
    heading_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*([^\n]*)", re.MULTILINE)

    # List to hold the extracted "full sections" as Document objects
    full_sections_docs = []
    
    # Buffers to accumulate content and page numbers for the current section
    current_section_lines = []
    current_section_pages_set = set()
    current_section_number = "N/A"
    current_full_section_heading = "General Policy/Introduction"

    # Helper to create a Document for the buffered section and reset buffers
    def finalize_and_add_section():
        nonlocal current_section_lines, current_section_pages_set, \
                   current_section_number, current_full_section_heading

        if current_section_lines:
            section_content = "\n".join(current_section_lines).strip()
            if section_content: # Only add if content is not empty
                full_sections_docs.append(
                    Document(
                        page_content=section_content,
                        metadata={
                            "source": filename,
                            "full_section_heading": current_full_section_heading,
                            "section_number": current_section_number,
                            "original_pages": sorted(list(current_section_pages_set))
                        }
                    )
                )
        # Reset for the next section
        current_section_lines = []
        current_section_pages_set = set()
        current_section_number = "N/A" # Reset, will be updated by next heading
        current_full_section_heading = "General Policy/Introduction" # Reset

    # --- Pass 1: Aggregate content into logical sections ---
    for page_content, page_num in page_contents_with_metadata:
        lines = page_content.split('\n')
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line: # Skip empty lines for heading detection
                current_section_lines.append(line) # Keep blank lines within content though
                continue

            match = heading_pattern.match(stripped_line)
            
            if match:
                # A new section heading is found
                finalize_and_add_section() # Finalize the previous section first
                
                # Update current section info
                current_section_number = match.group(1).strip()
                section_title = match.group(2).strip()
                current_full_section_heading = f"{current_section_number}. {section_title}" if section_title else current_section_number
                
                # Start buffering for the new section
                current_section_lines.append(line) # Add the heading line itself to the content
            else:
                # This line belongs to the current section
                current_section_lines.append(line)
            
            if page_num != 'N/A':
                current_section_pages_set.add(page_num)

    # Finalize the very last section in the document
    finalize_and_add_section()

    print(f"Manual {filename}: Identified {len(full_sections_docs)} distinct policy sections in Pass 1.")

    # --- Pass 2: Chunk these "full sections" using RecursiveCharacterTextSplitter ---
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Standard separators for breaking down section content
    )

    for section_doc in full_sections_docs:
        # Split the content of each *section* into smaller chunks
        chunks_from_this_section = text_splitter.split_text(section_doc.page_content)
        
        for chunk_text in chunks_from_this_section:
            # Create a new Document for each chunk, inheriting all metadata from the parent section
            new_chunk = Document(
                page_content=chunk_text,
                metadata={
                    "source": section_doc.metadata["source"],
                    "full_section_heading": section_doc.metadata["full_section_heading"],
                    "section_number": section_doc.metadata["section_number"],
                    "original_pages": section_doc.metadata["original_pages"] 
                }
            )
            final_chunks.append(new_chunk)

    print(f"Manual {filename}: Final chunking resulted in {len(final_chunks)} smaller chunks (Pass 2).")
    return final_chunks


def create_vector_store(chunks, embeddings_model):
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

# --- chunk_retrieval ---
def chunk_retrieval(relevant_chunks_by_manual_section, manual_vector_stores_dict, policy_area, top_k):
    # Extract the main section number from the policy_area query for filtering
    main_section_match = re.match(r"^(\d+(?:\.\d+)*)\.?", policy_area)
    target_section_prefix_for_filter = main_section_match.group(1) if main_section_match else None

    print(f"\n--- Retrieving Chunks for Policy Area: '{policy_area}' ---")
    print(f"Target section prefix for filtering: {target_section_prefix_for_filter}")

    for manual_name, vector_store in manual_vector_stores_dict.items():
        print(f"  Searching in {manual_name}...")
        # Step 1: Initial similarity search - retrieve a very large pool
        retrieved_docs_raw = vector_store.similarity_search(policy_area, k=top_k * 20) # Even more aggressive retrieval
        print(f"    Initial similarity search returned {len(retrieved_docs_raw)} chunks.")

        # Step 2: Filter by 'section_number' metadata
        filtered_docs_by_section = []
        if target_section_prefix_for_filter and target_section_prefix_for_filter != "N/A":
            for doc in retrieved_docs_raw:
                doc_section_number = doc.metadata.get("section_number", "N/A")
                if doc_section_number.startswith(target_section_prefix_for_filter):
                    filtered_docs_by_section.append(doc)
        else:
            filtered_docs_by_section = retrieved_docs_raw
        
        print(f"    After section filtering, {len(filtered_docs_by_section)} chunks remain.")

        # Step 3: Take the top_k relevant documents after section filtering
        final_docs_for_llm = filtered_docs_by_section[:top_k]

        if final_docs_for_llm:
            print(f"  Final selection: {len(final_docs_for_llm)} chunks from {manual_name}")
            for doc in final_docs_for_llm:
                section_num = doc.metadata.get('section_number', 'N/A')
                full_section_head = doc.metadata.get('full_section_heading', 'N/A')
                original_pages = doc.metadata.get('original_pages', []) 
                
                if manual_name not in relevant_chunks_by_manual_section:
                    relevant_chunks_by_manual_section[manual_name] = {}
                
                if section_num not in relevant_chunks_by_manual_section[manual_name]:
                    relevant_chunks_by_manual_section[manual_name][section_num] = {
                        'full_heading': full_section_head,
                        'content': [],
                        'page_info_set': set() # Use a set to track unique pages
                    }
                relevant_chunks_by_manual_section[manual_name][section_num]['content'].append(doc.page_content)
                
                for p_num in original_pages:
                    relevant_chunks_by_manual_section[manual_name][section_num]['page_info_set'].add(p_num)

        else:
            print(f"  No relevant chunks found in {manual_name} for '{policy_area}' after section filtering.")


# --- compile_and_compare_policy_area ---
def compile_and_compare_policy_area(policy_area, manual_vector_stores_dict, llm_model, top_k):
    print(f"\n--- Analyzing Policy Area: '{policy_area}' ---")
    
    relevant_chunks_by_manual_section = {}
    chunk_retrieval(relevant_chunks_by_manual_section, manual_vector_stores_dict, policy_area, top_k)

    if not relevant_chunks_by_manual_section:
        return {"context_text": "", "llm_response": f"No information found for '{policy_area}' across any of the manuals."}

    all_section_numbers = set()
    for manual_data in relevant_chunks_by_manual_section.values():
        all_section_numbers.update(manual_data.keys())

    def sort_key(s):
        return [int(part) if part.isdigit() else part for part in s.split('.')]
    
    sorted_section_numbers = sorted(list(all_section_numbers), key=sort_key)

    context_text_parts = []
    
    for section_num in sorted_section_numbers:
        # Get content for 2018
        content_2018 = None
        heading_2018 = None
        pages_2018_str = "N/A"
        if MANUAL_OLD_NAME in relevant_chunks_by_manual_section and \
           section_num in relevant_chunks_by_manual_section[MANUAL_OLD_NAME]:
            section_data_2018 = relevant_chunks_by_manual_section[MANUAL_OLD_NAME][section_num]
            content_2018 = "\n".join(section_data_2018['content']).strip()
            heading_2018 = section_data_2018['full_heading']
            pages_2018_str = "Page " + ", ".join(map(str, sorted(list(section_data_2018['page_info_set'])))) if section_data_2018['page_info_set'] else "N/A"


        # Get content for 2023
        content_2023 = None
        heading_2023 = None
        pages_2023_str = "N/A"
        if MANUAL_NEWEST_NAME in relevant_chunks_by_manual_section and \
           section_num in relevant_chunks_by_manual_section[MANUAL_NEWEST_NAME]:
            section_data_2023 = relevant_chunks_by_manual_section[MANUAL_NEWEST_NAME][section_num]
            content_2023 = "\n".join(section_data_2023['content']).strip()
            heading_2023 = section_data_2023['full_heading']
            pages_2023_str = "Page " + ", ".join(map(str, sorted(list(section_data_2023['page_info_set'])))) if section_data_2023['page_info_set'] else "N/A"

        # Append to context_text_parts in chronological order (2018 then 2023 if both exist)
        if content_2018:
            context_text_parts.append(
                f"--- Source: {MANUAL_OLD_NAME}, Page/Chunk: {pages_2018_str}, Section: {heading_2018} ({section_num}) ---\n"
                f"{content_2018}\n"
            )
        if content_2023:
            context_text_parts.append(
                f"--- Source: {MANUAL_NEWEST_NAME}, Page/Chunk: {pages_2023_str}, Section: {heading_2023} ({section_num}) ---\n"
                f"{content_2023}\n"
            )
    
    context_text = "\n\n".join(context_text_parts)

    prompt = f"""
    You are an expert HR policy analyst. Review and compare the **EXACT, VERBATUM** content from the provided excerpts of ROPSSA's employee manuals concerning the policy area: "{policy_area}".

    **YOUR OUTPUT MUST FOLLOW THIS STRUCTURE AND THESE CRUCIAL CONSTRAINTS (STRICTLY ADHERE):**

    **PART 1: Synthesized Policy - Verbatim**
    * Present **ALL** policy information related to "{policy_area}" from the provided excerpts.
    * **MUST** include **EVERY DISTINCT NUMBERED SECTION (e.g., 7., 7.1., 7.1.1., 7.1.1.1., 7.1.1.5.1., 7.1.1.5.2.)** that appears in the "Excerpts from Employee Manuals" section below.
    * List these sections in **STRICT NUMERICAL ORDER** (e.g., 7. then 7.1. then 7.1.1. then 7.1.1.1. then 7.1.1.2. then 7.1.1.3. then 7.1.1.3.1. then 7.1.1.5.1. then 7.1.1.5.2. etc.).
    * **For each listed section, START THE LINE WITH ITS EXACT SECTION NUMBER AND HEADING (e.g., "7. Nepotism:", "7.1. Definition:"). Then, provide its EXACT, VERBATIM text from the source manuals.**
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
    **Excerpts from Employee Manuals (FOR YOUR REFERENCE AND VERBATUM COPY-PASTE):**
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
    
    font_dir = "./fonts"
    if not os.path.exists(font_dir):
        print(f"Warning: Font directory '{font_dir}' not found. PDF report might use default fonts.")
    else:
        required_fonts = ["NotoSans-Regular.ttf", "NotoSans-Italic.ttf", "NotoSans-Bold.ttf", "NotoSans-BoldItalic.ttf"]
        for font_file in required_fonts:
            if not os.path.exists(os.path.join(font_dir, font_file)):
                print(f"Warning: Required font file '{font_file}' not found in '{font_dir}'. PDF report might use default fonts.")

    # Load entire document content and page mapping
    page_contents_O, filename_O = load_document_content(os.path.join(DOCUMENT_LIBRARY_PATH, MANUAL_OLD_NAME))
    page_contents_N, filename_N = load_document_content(os.path.join(DOCUMENT_LIBRARY_PATH, MANUAL_NEWEST_NAME))

    if not page_contents_O or not page_contents_N:
        print("Error: Could not load full text from one or more employee manuals. Please check paths and content. Exiting.")
        exit()

    # Split documents into chunks for vector store creation, passing page_offsets
    # Chunk size and overlap apply to the *secondary* chunking within identified sections
    chunk_size = 200 # Made even smaller to ensure very fine-grained chunks for short policies
    chunk_overlap = 50 # Adjusted overlap
    manual_O_chunks = split_documents(page_contents_O, filename_O, chunk_size, chunk_overlap)
    manual_N_chunks = split_documents(page_contents_N, filename_N, chunk_size, chunk_overlap)

    print(f"Manual {filename_O} split into {len(manual_O_chunks)} chunks.")
    print(f"Manual {filename_N} split into {len(manual_N_chunks)} chunks.")


    # Create individual vector stores for each manual for targeted retrieval
    manual_vector_stores = {
        filename_O: create_vector_store(manual_O_chunks, embeddings),
        filename_N: create_vector_store(manual_N_chunks, embeddings)
    }
    print("\nAll manuals loaded and individual vector stores created.")

    compiled_report_by_topic = {}
    print("\n--- Starting Compilation and Contradiction Detection ---")

    # Kept top_k high, as it's the number of final chunks sent to LLM after filtering
    top_k = 20 
    for topic in KEY_POLICY_AREAS:
        compiled_data = compile_and_compare_policy_area(topic, manual_vector_stores, llm, top_k)
        compiled_report_by_topic[topic] = compiled_data
        print(f"\n{'='*20} End of Analysis for '{topic}' {'='*20}\n")

    print("\n--- Full Compiled Report (Console Preview) ---")
    for topic, data in compiled_report_by_topic.items():
        print(f"\n### Policy Area: {topic}\n")
        print(data["llm_response"])
        print("\n--------------------------------------------------")

    output_directory = "./output_reports_version_9" # New output directory
    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_filename = os.path.join(output_directory, f"report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_report_by_topic, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")