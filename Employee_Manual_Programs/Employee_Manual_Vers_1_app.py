import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document # Import Document class explicitly
import json # To pretty print extracted schema data

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./dummy_manuals" # Create a folder named 'dummy_manuals' for your documents

# Names of your three employee manuals
# MAKE SURE THESE FILENAMES MATCH YOUR ACTUAL FILES IN THE 'dummy_manuals' FOLDER
MANUAL_OLD_OLD_NAME = "employee_manual_v1_old_old.txt" # The oldest version
MANUAL_OLD_NAME = "employee_manual_v2_old.txt"      # The middle version
MANUAL_NEWEST_NAME = "employee_manual_v3_newest.txt" # The newest version

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "gemma3:latest"
# dedicated embedding model that supports embeddings
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Define key policy areas/topics to analyze.
# You can expand or modify this list based on what's important in your manuals.
KEY_POLICY_AREAS = [
    "Company Values and Culture",
    "Working Hours and Attendance",
    "Sick Leave Policy",
    "Vacation Policy",
    "Public Holidays",
    "Compensation and Payroll",
    "Benefits (Health, Retirement, etc.)",
    "Code of Conduct and Ethics",
    "Anti-Discrimination and Equal Opportunity",
    "Data Privacy and Confidentiality",
    "Use of Company Property",
    "Performance Reviews and Management",
    "Disciplinary Procedures",
    "Grievance and Complaint Procedures",
    "Termination and Resignation Procedures",
    "Employee Feedback Mechanisms",
    "Training and Development",
    "Safety and Health Policy"
]

# --- Initialization ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=0.1)

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
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    # Splits large documents into smaller chunks for LLM processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
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
    You are an expert HR policy analyst. Your task is to review excerpts from different versions
    of Ropsa's employee manuals concerning the policy area: "{policy_area}".

    **Instructions:**
    1.  **Synthesize all consistent information** regarding "{policy_area}" from the provided excerpts. Present this as the general policy.
    2.  **Identify and list any contradictions or significant changes** in policy details or wording between the different manual versions for "{policy_area}".
    3.  For each contradiction or change, **explicitly quote the relevant text from each conflicting manual version** and clearly state the source manual (e.g., 'employee_manual_v1_old_old.txt', 'employee_manual_v2_old.txt', 'employee_manual_v3_newest.txt'). Present these instances one after the other.
    4.  If there are no contradictions and the information is consistent, simply state the synthesized policy.

    **Policy Area to Analyze:** {policy_area}

    **Excerpts from Employee Manuals:**
    ---
    {context_text}
    ---

    **Your Compiled Information and Analysis (Start with Overall Policy, then list contradictions):**
    """

    try:
        response = llm_model.invoke(prompt)
        return response
    except Exception as e:
        return f"Error processing '{policy_area}': {e}"


# --- Main Program Logic ---
    
# -- Helper Functions for Main --
    
def create_dummies():
    # Create dummy manuals if they don't exist
    dummy_manuals_content = {
        MANUAL_OLD_OLD_NAME: """
        Employee Manual V1 (Old Old) - ROPSSA
        1. Working Hours: Standard working hours are 9 AM to 5 PM, Monday to Friday. Lunch break is 30 minutes.
        2. Sick Leave: Employees accrue 1 day of sick leave per month, up to 10 days per year. No carryover. Doctor's note required after 2 days.
        3. Vacation: 10 days per year for all employees, after 1 year of service. Max 5 days carryover.
        4. Code of Conduct: Employees are expected to be professional. No specific anti-discrimination clause.
        5. Termination: 2 weeks notice for voluntary resignation. Company gives 2 weeks notice for termination.
        6. Data Privacy: Employee data is kept confidential internally.
        """,
        MANUAL_OLD_NAME: """
        Employee Manual V2 (Old) - ROPSSA
        1. Working Hours: Standard working hours are 9 AM to 5 PM, Monday to Friday. Lunch break is 1 hour.
        2. Sick Leave: Employees accrue 1.25 days of sick leave per month, up to 15 days per year. Max 5 days carryover. Doctor's note required for any absence.
        3. Vacation: 15 days per year for all employees, from start date. Max 5 days carryover. Requests require 2 weeks notice.
        4. Code of Conduct: Ropsa is committed to a harassment-free workplace. All employees must adhere to professional conduct. No explicit anti-discrimination.
        5. Termination: 2 weeks notice for voluntary resignation. Company gives 4 weeks notice for termination.
        6. Data Privacy: All employee personal data is handled according to internal Ropsa guidelines. Data may be shared with third-party service providers for payroll purposes only.
        """,
        MANUAL_NEWEST_NAME: """
        Employee Manual V3 (Newest) - ROPSSA
        1. Working Hours: Flexible working hours 8 AM to 6 PM, core hours 10 AM to 3 PM. Lunch break is 1 hour, unpaid.
        2. Sick Leave: Employees accrue 1.5 days of sick leave per month, up to 18 days per year. Max 10 days carryover. No doctor's note needed for first 3 consecutive days of absence.
        3. Vacation: 20 days per year for all employees, from start date. Max 10 days carryover. Requests require 4 weeks notice.
        4. Code of Conduct: Ropsa strictly prohibits discrimination based on age, gender, race, religion, sexual orientation, or disability. All employees are expected to maintain the highest level of professionalism and respect.
        5. Termination: 4 weeks notice required for both voluntary resignation and company-initiated termination.
        6. Data Privacy: Ropsa processes employee personal data in compliance with GDPR principles, ensuring data minimization, security, and consent. Employees have the right to access and rectify their data. Data is only shared with authorized third parties strictly for operational necessities and with clear consent.
        """
    }

    # Write dummy content to files if they don't exist
    for filename, content in dummy_manuals_content.items():
        filepath = os.path.join(DOCUMENT_LIBRARY_PATH, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content.strip())
            print(f"Created dummy file: {filepath}")


if __name__ == "__main__":
    # Ensure the directory exists
    if not os.path.exists(DOCUMENT_LIBRARY_PATH):
        os.makedirs(DOCUMENT_LIBRARY_PATH)
        print(f"Created directory: {DOCUMENT_LIBRARY_PATH}")

    # Create dummy documents for demonstration if they don't exist (essentially this is just to make sure we don't run huge errors if the files aren't there lemme protect my comptuer)

    create_dummies()

    # Load and Process Manuals
    manual_OO_docs = load_documents(DOCUMENT_LIBRARY_PATH, MANUAL_OLD_OLD_NAME)
    manual_O_docs = load_documents(DOCUMENT_LIBRARY_PATH, MANUAL_OLD_NAME)
    manual_N_docs = load_documents(DOCUMENT_LIBRARY_PATH, MANUAL_NEWEST_NAME)

    if not manual_OO_docs or not manual_O_docs or not manual_N_docs:
        print("Error: One or more employee manuals not found or empty. Please check paths and content. Exiting.")
        exit()

    # Split documents into chunks for vector store creation
    manual_OO_chunks = split_documents(manual_OO_docs, chunk_size=500, chunk_overlap=100)
    manual_O_chunks = split_documents(manual_O_docs, chunk_size=500, chunk_overlap=100)
    manual_N_chunks = split_documents(manual_N_docs, chunk_size=500, chunk_overlap=100)

    # Create individual vector stores for each manual for targeted retrieval
    manual_vector_stores = {
        MANUAL_OLD_OLD_NAME: create_vector_store(manual_OO_chunks, embeddings),
        MANUAL_OLD_NAME: create_vector_store(manual_O_chunks, embeddings),
        MANUAL_NEWEST_NAME: create_vector_store(manual_N_chunks, embeddings)
    }
    print("\nAll manuals loaded and individual vector stores created.")

    # --- Main Compilation and Contradiction Detection ---
    compiled_report_by_topic = {}
    print("\n--- Starting Compilation and Contradiction Detection ---")

    top_k = 5 # Retrieve top 5 relevant chunks from each manual for this policy area
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

    print("\nProgram Finished.")