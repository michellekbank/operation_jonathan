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
GUIDELINES_DOC_NAME = "PNC.txt" # Name of your guidelines document
EMPLOYEE_MANUAL_A_NAME = "employee_manual_A.txt" # Name of your guidelines document
EMPLOYEE_MANUAL_B_NAME = "employee_manual_B.txt" # Name of your guidelines document

# --- OLLAMA MODEL SETUP ---
OLLAMA_LLM_MODEL = "gemma3:latest"
OLLAMA_EMBEDDING_MODEL = "gemma3:latest" # Ollama models often provide embeddings from the same model


# Define extraction schemas for different document types or tasks
# Customize this schema based on what you need to extract from your employee manuals
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

# Define specific aspects/questions for targeted compliance checks
# These are broader themes that the LLM will focus on when comparing to guidelines
COMPLIANCE_ASPECTS_TO_CHECK = [
   "list all the stuff that we should check :))))"
]

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

def check_manual_compliance(employee_manual_content, guidelines_vectorstore, llm_model, compliance_aspects):
    #CHecks the compliance of an employee manual against guidelines from a vector store,
    #focusing on specific aspects.

    print("\n--- Checking Employee Manual Compliance ---")
    compliance_report = {}

    # Option 1: Iterate through all guideline chunks and ask for compliance for each
    # This retrieves the raw Document objects from the FAISS docstore
    # Note: For very large guideline documents, this might be slow or hit token limits.
    # Consider only searching for relevant guidelines per overall manual or by specific aspect.
    guideline_docs_from_store = list(guidelines_vectorstore.docstore._docs.values())

    print(f"Performing detailed compliance check against {len(guideline_docs_from_store)} guideline chunks...")
    for i, guideline_doc in enumerate(guideline_docs_from_store):
        guideline_text = guideline_doc.page_content
        guideline_source = guideline_doc.metadata.get('source', 'Unknown')
        
        prompt_specific = f"""
        You are a highly analytical compliance officer. Your task is to evaluate whether the
        'Employee Manual Content' explicitly complies with the 'Specific Guideline' provided.
        Be precise and objective.

        Employee Manual Content:
        ---
        {employee_manual_content}
        ---

        Specific Guideline (from {guideline_source}):
        ---
        {guideline_text}
        ---

        Does the employee manual comply with this specific guideline?
        Respond with 'COMPLIANT', 'NON-COMPLIANT', or 'PARTIALLY COMPLIANT'.
        Provide a concise, direct explanation, citing specific parts of the manual and guideline if applicable,
        or stating if the guideline is not addressed.
        """
        response = llm_model.invoke(prompt_specific)
        compliance_report[f"Guideline {i+1} from {guideline_source}"] = response
        print(f"  Checked Guideline {i+1} from {guideline_source}: {response.splitlines()[0]}...") # Print first line of response

    # Option 2: Check for specific compliance aspects using targeted queries
    if compliance_aspects:
        print("\n--- Checking specific compliance aspects with targeted queries ---")
        for aspect in compliance_aspects:
            # Retrieve relevant sections from the guidelines document based on the aspect query
            relevant_guidelines = guidelines_vectorstore.similarity_search(aspect, k=3) # Get top 3 relevant guideline chunks
            guidelines_context = "\n".join([f"Guideline: {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})" for doc in relevant_guidelines])

            prompt_aspect = f"""
            You are a compliance expert. Evaluate the following employee manual against the provided relevant guidelines
            specifically regarding the aspect: "{aspect}".

            Employee Manual Content:
            ---
            {employee_manual_content}
            ---

            Relevant Guidelines (related to '{aspect}'):
            ---
            {guidelines_context}
            ---

            Does the employee manual comply with the guidelines regarding "{aspect}"?
            Respond with 'COMPLIANT', 'NON-COMPLIANT', or 'PARTIALLY COMPLIANT'.
            Explain your reasoning clearly, citing relevant parts of the manual and guidelines where possible.
            """
            response = llm_model.invoke(prompt_aspect)
            compliance_report[f"Aspect: {aspect}"] = response
            print(f"  Checked Aspect '{aspect}': {response.splitlines()[0]}...")

    return compliance_report