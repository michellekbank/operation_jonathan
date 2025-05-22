import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama # For LLM
from langchain_community.embeddings import OllamaEmbeddings # For Embeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
from langchain.docstore.document import Document # Import Document class explicitly


# --- SETUP ---

DOCUMENT_LIBRARY_PATH = "./my_documents"
EMPLOYEE_MANUAL_A_NAME = "employee_manual_A.txt" # Name of your guidelines document
EMPLOYEE_MANUAL_B_NAME = "employee_manual_B.txt" # Name of your guidelines document
EMPLOYEE_MANUAL_C_NAME = "employee_manual_C.txt"

# --- OLLAMA MODEL SETUP ---
OLLAMA_LLM_MODEL = "gemma3:latest"
OLLAMA_EMBEDDING_MODEL = "gemma3:latest" # Ollama models often provide embeddings from the same model

EMPLOYEE_MANUAL_SCHEMA = {
    "properties": {
        "manual_title": {"type": "string", "description": "The official title of the employee manual."},
        "company_name": {"type": "string", "description": "The name of the company this manual belongs to."},
        "sick_leave_policy": {"type": "string", "description": "A summary of the sick leave policy."},
        "vacation_policy": {"type": "string", "description": "A summary of the vacation leave policy."},
        "anti_discrimination_statement": {"type": "string", "description": "The explicit statement about anti-discrimination."}
    },
    "required": ["manual_title", "company_name"]
}

# --- INITIALIZATION ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = Ollama(model=OLLAMA_LLM_MODEL, temperature=0)

print(f"Initializing Embeddings with Ollama model: {OLLAMA_EMBEDDING_MODEL}")
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

# -- FUNCTIONS --

def load_documents(directory_path, filename=None):
    #Loads documents from a given directory or a specific file
    documents = []
    if filename:
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
        for doc in docs:
            doc.metadata["source"] = filename # Add original filename as metadata
        documents.extend(docs)
    else:
        for filename_in_dir in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename_in_dir)
            if filename_in_dir.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename_in_dir.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
            elif filename_in_dir.endswith(".txt"):
                loader = TextLoader(filepath)
            else:
                print(f"Skipping unsupported file type: {filename_in_dir}")
                continue
            print(f"Loading {filename_in_dir}...")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename_in_dir
            documents.extend(docs)
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    #Splits large documents into smaller chunks for LLM processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks, embeddings_model):
    #Creates a FAISS vector store from document chunks and embeddings
    print("Creating vector store...")
    return FAISS.from_documents(chunks, embeddings_model)

def compare_documents_semantic(doc1_content, doc2_content, llm_model, embeddings_model, top_k=3):
    # Compares two document contents semantically by finding similar chunks
    # and asking the LLM to identify similarities/differences
    print("\nPerforming semantic comparison...")
    # Ensure doc1_content and doc2_content are treated as LangChain Document objects
    # for consistent processing by split_documents
    doc1_chunks = split_documents([Document(page_content=doc1_content)])
    doc2_chunks = split_documents([Document(page_content=doc2_content)])

    vectorstore1 = create_vector_store(doc1_chunks, embeddings_model)
    vectorstore2 = create_vector_store(doc2_chunks, embeddings_model)

    similarities = []
    for chunk1 in doc1_chunks:
        # Retrieve semantically similar chunks from doc2's vector store
        retrieved_chunks = vectorstore2.similarity_search(chunk1.page_content, k=top_k)
        for chunk2 in retrieved_chunks:
            prompt = f"""
            Compare the following two text excerpts and identify their key similarities and differences.
            Excerpt 1: "{chunk1.page_content}"
            Excerpt 2: "{chunk2.page_content}"
            """
            response = llm_model.invoke(prompt)
            similarities.append({
                "doc1_chunk": chunk1.page_content,
                "doc2_chunk": chunk2.page_content,
                "comparison_summary": response
            })
    return similarities

def summarize_documents(documents, llm_model, summary_type="stuff"):
    #Summarizes a list of LangChain Document objects
    print(f"\nSummarizing documents with '{summary_type}' chain...")
    chain = load_summarize_chain(llm_model, chain_type=summary_type)
    # The invoke method of summarization chain expects a list of Document objects
    summary = chain.invoke(documents)
    return summary['output_text'] # Access the summary text

def extract_information(document_content, llm_model, schema):
    #Extracts structured information from document content based on a schema
    print("\nExtracting information...")
    chain = create_extraction_chain(schema, llm_model)
    # Ensure document_content is a string for the extraction chain's invoke method
    if isinstance(document_content, Document):
        document_content = document_content.page_content
    extracted_data = chain.invoke({"input": document_content}) # Pass content as dictionary for extraction chain
    return extracted_data['text'] # The extraction chain usually returns a dict with a 'text' key