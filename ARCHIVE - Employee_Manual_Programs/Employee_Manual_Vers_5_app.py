# CHANGES FROM VERSION 4
# 1. Key policy areas back to what they were
# 2. Change the LLM model to granite3-dense
# 3. Print context text and response
# 4. Experiment with LLM model granite3.2:2b

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document # Import Document class explicitly
from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./employee_manuals"

# Names of your two employee manuals
# MAKE SURE THESE FILENAMES MATCH YOUR ACTUAL FILES IN THE 'employee_manuals' FOLDER
MANUAL_OLD_NAME = "Employee_Manual_2018.docx"      # The middle version
MANUAL_NEWEST_NAME = "Employee_Manual_2023.pdf" # The newest version

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "granite3.2:2b"
# dedicated embedding model that supports embeddings
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Define key policy areas/topics to analyze.
# You can expand or modify this list based on what's important in your manuals.
KEY_POLICY_AREAS = ["7. Employment"]

# --- Initialization ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.0)

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL) 
# Ollama kinda goated because we can do the embeddings right here :)

# --- Functions ---


# -- Helper functions for load_document --

# given a Document object list, we add in useful metadata
def add_metadata(docs, filename):
    for i, doc in enumerate(docs):
        # Add original filename and page number (if applicable) as metadata
        doc.metadata["source"] = filename
        if 'page' in doc.metadata: # PyPDFLoader adds 'page'
            doc.metadata["original_page"] = doc.metadata['page']
        else: # For txt/docx, we might simulate page numbers or just note the chunk number
            doc.metadata["original_page"] = f"chunk_{i+1}" # Simple chunk indicator


# Loads the given file if one is provided, modifying the "documents" list to
# include information from the given file
def load_filename(documents, directory_path, filename):
    filepath = os.path.join(directory_path, filename)

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    # gives us a LangChain loader corresponding with whatever filetype the document is
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath) 
    elif filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        print(f"Skipping unsupported file type: {filename}")
        return []
    
    # reads the content from the document and puts it in a LangChain Document object list
    print(f"Loading {filename}...")
    docs = loader.load()
    add_metadata(docs, filename)
    documents.extend(docs)

# in the case that no file is provided to load_documents, we simply load all of 
#    the files in the directory

def load_directory(documents, directory_path):
    # loops through the files in the directory path and loads them 
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
                    add_metadata(docs, filename_in_dir)
                    documents.extend(docs)
            else:
                print(f"Skipping unsupported file type: {filename_in_dir}")

# -- load_documents --
#  Loads documents from a given directory or a specific file
def load_documents(directory_path, filename=None):
    documents = []
    if filename:
        load_filename(documents, directory_path, filename)
    else: # Load all documents in a directory
        load_directory(documents, directory_path)
    return documents

# -- split_documents --
#   uses a recursive character text splitter to take the documents and split 
#   them in a way that preserves semantics. This ensures that it can be processed
#   by the llm
def split_documents(documents, chunk_size, chunk_overlap):
    # Splits large documents into smaller chunks for LLM processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            r"\d+\.\d+\.\d+\.\d+\.\d+\.\s", # Regex for 7.1.1.1.1 (most most specific)
            r"\d+\.\d+\.\d+\.\d+\.\s",     # Regex for 7.1.1.1 (specific)
            r"\d+\.\d+\.\d+\.\s",          # Regex for 7.1.1. (more specific mid-level)
            r"\d+\.\d+\.\s",               # Regex for 7.1. (mid-level)
            r"\d+\.\s",                    # Regex for 7. (top-level)
            "\n\n",                        # Try to split by double newline (paragraphs)
            "\n",                          # Then single newline (lines)
            " ",                           # Then by space (words)
            "",                            # Fallback to characters
        ]
    )
    return text_splitter.split_documents(documents)

# -- create_vector_store
# Creates a FAISS vector store from document chunks and embeddings :)
def create_vector_store(chunks, embeddings_model):
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

# -- Helper functions for compile_and_compare_policy_area --

# Retrieve chunks relevant to the policy area from each manual's vector store
def chunk_retrieval(relevant_chunks_for_llm, manual_vector_stores_dict, policy_area, top_k):
    for manual_name, vector_store in manual_vector_stores_dict.items():
        retrieved_docs = vector_store.similarity_search(policy_area, k=top_k)
        if retrieved_docs:
            print(f"  Retrieved {len(retrieved_docs)} chunks from {manual_name}")
            for doc in retrieved_docs:
                # Format each retrieved chunk with its source manual and (simulated) page/chunk info
                relevant_chunks_for_llm.append(
                    f"--- Source: {manual_name}, Page/Chunk: {doc.metadata.get('original_page', 'N/A')} ---\n"
                    f"{doc.page_content}\n"
                )
        else:
            print(f"  No relevant chunks found in {manual_name} for '{policy_area}'.")

# -- compile_and_compare_policy_area --
#      Compiles information for a given policy area across multiple manuals 
#      and highlights contradictions or significant changes.
def compile_and_compare_policy_area(policy_area, manual_vector_stores_dict, # Dictionary that maps manual name to its vector store
    llm_model, top_k):
    print(f"\n--- Analyzing Policy Area: '{policy_area}' ---")
    relevant_chunks_for_llm = []

    chunk_retrieval(relevant_chunks_for_llm, manual_vector_stores_dict, policy_area, top_k)

    if not relevant_chunks_for_llm:
        return f"No information found for '{policy_area}' across any of the manuals."

    context_text = "\n\n".join(relevant_chunks_for_llm)

    prompt = f"""
    You are an expert HR policy analyst. Your task is to meticulously review and compare **ONLY** the provided excerpts from two different versions
    of ROPSSA's employee manuals concerning the policy area: "{policy_area}".

    **CRUCIAL CONSTRAINTS - READ CAREFULLY. STRICT ADHERENCE REQUIRED:**
    * **DO NOT REPHRASE, SUMMARIZE, OR ABBREVIATE** any policy text in PART 1. You **MUST** use the exact wording from the provided "Excerpts from Employee Manuals."
    * **DO NOT ADD ANY NEW INFORMATION, SECTIONS, SUB-SECTIONS, OR HEADINGS** that are not explicitly present in the provided "Excerpts from Employee Manuals."
    * **ALL INFORMATION, SECTION NUMBERS, AND HEADINGS** in your output **MUST BE DIRECTLY QUOTED OR EXACTLY DERIVED** from the provided excerpts.
    * **DO NOT INFER, CONCLUDE, OR GENERATE CONTENT BASED ON EXTERNAL KNOWLEDGE.** Stick strictly to the given text.
    * **DO NOT OMMIT ANY INFORMATION. ** it is crucial that all information from the provided excerpts is included in your response.

    ---
    ### PART 1: SYNTHESIZED CONSISTENT POLICY (EXACT QUOTES ONLY)

    1.  **Synthesize and Combine (Verbatim):** Combine all consistent information regarding "{policy_area}" **EXCLUSIVELY** from the provided manuals. For every piece of information, you **MUST copy and paste the exact, full sentence(s) or paragraph(s)** from the source text.
    2.  **Include Section Numbers & Headings (Exactly as Found):** For all synthesized policy details, you **MUST** include the exact section numbers (e.g., "7.1.1. Application") and their corresponding headings/titles **as found in the original excerpts**.
        * **Ensure you include information from ALL appropriate section numbers (e.g., 8.1 through 8.9) that are present in the provided excerpts.**
        * **DO NOT invent or re-number any sections that are not explicitly in the source text.**
    3.  **Prioritize Newest Manual Language (Verbatim):** When identical or nearly identical content exists across manuals, prioritize and **use the exact language and structure from the 2023 manual**. Copy it verbatim.

    ---
    ### PART 2: CONTRADICTIONS AND SIGNIFICANT CHANGES

    If **ANY** contradictions or significant changes in policy are identified **within the provided excerpts**, list them clearly. If there are none, state "No contradictions or significant changes identified for this policy area."

    For EACH contradiction or change, follow this exact format:

    **Contradiction/Change in [Specific Policy Aspect, e.g., 'Vacation Accrual Rate']:**
    * **[2018 Manual - Section Number & Heading]:** "[Exact Quote from 2018 Manual]"
    * **[2023 Manual - Section Number & Heading]:** "[Exact Quote from 2023 Manual]"
    * **Significance:** [Brief explanation of the change/contradiction and its implications, **based ONLY on the provided texts**.]

    ---

    **Policy Area to Analyze:** {policy_area}

    **Excerpts from Employee Manuals (including source, page/chunk, and section context):**
    ---
    {context_text}
    ---

    **Your Compiled Information and Analysis (PART 1, then PART 2):**
    """

    try:
        response = llm_model.invoke(prompt)
        return {"context_text": context_text, "llm_response": response}
    except Exception as e:
        return f"Error processing '{policy_area}': {e}"


# -- Helper Function to write report to a PDF --
    
# Writes the compiled policy report to a PDF file.
    
def write_report_to_pdf(report_data, output_filepath):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15) # Enable auto page breaks with a margin
    
    # --- FONT DEFINITIONS ---

    font_dir = "./fonts"
    regular_font_path = os.path.join(font_dir, "NotoSans-Regular.ttf")
    italic_font_path = os.path.join(font_dir, "NotoSans-Italic.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSans-Bold.ttf")
    bolditalic_font_path = os.path.join(font_dir, "NotoSans-BoldItalic.ttf")


    pdf.add_font("NotoSans", "", regular_font_path)
    pdf.add_font("NotoSans", "I", italic_font_path) 
    pdf.add_font("NotoSans", "B", bold_font_path) 
    pdf.add_font("NotoSans", "BI", bolditalic_font_path) 

    # --- END FONT DEFINITIONS ---

    # Add a title page or general header
    pdf.add_page()
    pdf.set_font("NotoSans", "BI", 20)
    pdf.multi_cell(0, 12, "ROPSSA Employee Manuals - Compiled Policy Report", align='C')
    pdf.ln(15) # Line break

    pdf.set_font("NotoSans", "", 12)
    pdf.multi_cell(0, 8, "This report synthesizes information from various versions of ROPSSA's employee manuals and highlights any contradictions or significant changes identified by the LLM.")
    pdf.ln(10)

    for topic, data in report_data.items(): # 'data' now holds the dictionary: {"llm_response": ..., "context_text": ...}
        llm_response_content = data["llm_response"] # Get the LLM's response string
        context_text_content = data["context_text"] # Get the raw context string

        # Main Policy Area Heading
        if pdf.get_y() > (pdf.h - 60): # Added more margin for the new section
            pdf.add_page()
        pdf.set_font("NotoSans", "B", 16)
        pdf.multi_cell(0, 10, f"Policy Area: {topic}", 0, 'L')
        pdf.ln(4)

        # LLM's Compiled Information
        pdf.set_font("NotoSans", "", 11) # Regular font for content
        # Use the variable holding the string content
        pdf.multi_cell(0, 6, llm_response_content.strip()) # <--- FIXED HERE
        pdf.ln(8) # Space after content

        # Raw Context Text Section
        if context_text_content.strip(): # Only add if there's actual context
            if pdf.get_y() > (pdf.h - 40): # Check for page break before adding context
                pdf.add_page()
            pdf.set_font("NotoSans", "B", 12)
            pdf.multi_cell(0, 8, "--- Raw Context Provided to LLM ---", 0, 'L')
            pdf.ln(2)
            pdf.set_font("NotoSans", "I", 9) # Smaller, italic font for raw context
            # Use the variable holding the string content
            pdf.multi_cell(0, 4, context_text_content.strip()) # <--- FIXED HERE
            pdf.ln(8)

        # Add a visual separator for clarity between policy areas
        pdf.set_font("NotoSans", "I", 9) # Italic, smaller font for separator
        pdf.multi_cell(0, 5, "-" * 100, align='C') # Centered dashes
        pdf.ln(8)


    pdf.output(output_filepath)


# --- Main Program Logic ---


if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists(DOCUMENT_LIBRARY_PATH):
        os.makedirs(DOCUMENT_LIBRARY_PATH)
        print(f"Created directory: {DOCUMENT_LIBRARY_PATH}")

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

    # --- Main Compilation and Contradiction Detection ---
    compiled_report_by_topic = {}
    print("\n--- Starting Compilation and Contradiction Detection ---")

    top_k = 7 # Retrieve top 5 relevant chunks from each manual for this policy area
    for topic in KEY_POLICY_AREAS:
        # Pass the dictionary of vector stores to the compilation function
        compiled_info = compile_and_compare_policy_area(topic, manual_vector_stores, llm, top_k)
        compiled_report_by_topic[topic] = compiled_info
        print(f"\n{'='*20} End of Analysis for '{topic}' {'='*20}\n")

    print("\n--- Full Compiled Report ---")
    for topic, info in compiled_report_by_topic.items():
        print(f"\n### Policy Area: {topic}\n")
        print(info)
        print("\n--------------------------------------------------")

    output_directory = "./output_reports_version_5"
    os.makedirs(output_directory, exist_ok=True) # Ensure the output directory exists

    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # this saves the report to a pdf in the appropriate folder
    pdf_output_filename = os.path.join(output_directory, f"report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_report_by_topic, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")