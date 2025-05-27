# CHANGES FROM VERSION 9
# this is just a sanity check to make sure we can't just feed it all into the LLM and
# ask it to do our job for us...

import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_ollama import OllamaLLM
from fpdf import FPDF
from datetime import datetime

# --- Configuration ---
DOCUMENT_LIBRARY_PATH = "./employee_manuals"

MANUAL_OLD_NAME = "Employee_Manual_2018.pdf"
MANUAL_NEWEST_NAME = "Employee_Manual_2023.pdf"

# --- Ollama Model Configuration ---
OLLAMA_LLM_MODEL = "granite3-dense"

# Define key policy areas/topics to analyze.
KEY_POLICY_AREAS = [
   "24. Nepotism"
]

# --- Initialization ---
print(f"Initializing LLM with Ollama model: {OLLAMA_LLM_MODEL}")
llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.1)


# --- Functions ---

    # Loads the entire raw text content from a single document.
    # Returns the concatenated text from all pages.  
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
        return None

    print(f"Loading raw content from {os.path.basename(filepath)}...")
    raw_docs = loader.load()
    
    # Concatenate all page contents into a single string
    full_text = "\n\n".join([doc.page_content for doc in raw_docs])
    
    return full_text

# --- compile_and_compare_policy_area ---
def compile_and_compare_policy_area(policy_area, manual_old_text, manual_newest_text, llm_model):
    """
    Compiles policy information and identifies contradictions using an LLM,
    directly from raw document texts.
    """
    print(f"\n--- Analyzing Policy Area: '{policy_area}' ---")
    
    # Combine the raw texts from both manuals for the LLM's context
    context_text = (
        f"--- Content from {MANUAL_OLD_NAME} ---\n"
        f"{manual_old_text}\n\n"
        f"--- Content from {MANUAL_NEWEST_NAME} ---\n"
        f"{manual_newest_text}"
    )

    prompt = f"""
    You are an expert HR policy analyst. Your task is to meticulously review and compare the content
    from the provided raw text of ROPSSA's employee manuals concerning the policy area: "{policy_area}".

    **CRUCIAL CONSTRAINTS (STRICTLY ADHERE):**

    **PART 1: Synthesized Policy - Verbatim**
    * **IDENTIFY AND EXTRACT** all policy information related to "{policy_area}" from the provided raw texts.
    * **MUST** include **EVERY DISTINCT NUMBERED SECTION (e.g., 24., 24.1., 24.2., etc.)** that you identify as relevant to "{policy_area}".
    * List these sections in **STRICT NUMERICAL ORDER** (e.g., 24. then 24.1. then 24.2. etc.).
    * **For each listed section, START THE LINE WITH ITS EXACT SECTION NUMBER AND HEADING (e.g., "24. Nepotism:", "24.1. The hiring of persons:"). Then, provide its EXACT, VERBATIM text from the source manuals.**
    * If a section's content is identical or nearly identical across both manuals, **PRIORITIZE** and use the verbatim text from the **2023 manual**.
    * **DO NOT** rephrase, summarize, abbreviate, or add any new information.
    * **DO NOT** omit any information from the relevant sections.
    * **DO NOT** add any explanatory text, comments, analysis, or any other information not explicitly present in the provided text for PART 1.
    * **DO NOT** mention "This policy remains unchanged" or similar phrases.

    **PART 2: Contradictions & Significant Changes**
    * Analyze the "Raw Manual Contents" provided below to identify **ALL** contradictions or significant changes between the 2018 and 2023 manuals for the "{policy_area}" policy.
    * If no contradictions or significant changes are identified, state: "No contradictions or significant changes identified."
    * **YOU MUST USE THIS EXACT FORMAT FOR EACH CHANGE. DO NOT DEVIATE. If there are no changes, just state the "No contradictions..." sentence.**
    * For each change, use this **EXACT FORMAT**:
        **Contradiction/Change in [Specific Policy Aspect]:**
        * **[2018 Manual - Section No. & Heading]:** "[Exact Quote from 2018 Manual]"
        * **[2023 Manual - Section No. & Heading]:** "[Exact Quote from 2023 Manual]"
        * **Significance:** [Brief explanation **based ONLY on provided texts**, explaining the nature of the difference.]

    ---
    **Raw Manual Contents (FOR YOUR REFERENCE AND VERBATUM COPY-PASTE):**
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

    try:
        if os.path.exists(regular_font_path):
            pdf.add_font("NotoSans", "", regular_font_path)
            pdf.set_font("NotoSans", "", 12)
        else:
            print(f"Warning: NotoSans-Regular.ttf not found at {regular_font_path}. Using default font.")
            pdf.set_font("Helvetica", "", 12)
        if os.path.exists(italic_font_path):
            pdf.add_font("NotoSans", "I", italic_font_path)
        if os.path.exists(bold_font_path):
            pdf.add_font("NotoSans", "B", bold_font_path)
        if os.path.exists(bolditalic_font_path):
            pdf.add_font("NotoSans", "BI", bolditalic_font_path)
    except Exception as e:
        print(f"Error loading custom fonts: {e}. Falling back to default fonts.")
        pdf.set_font("Helvetica", "", 12)


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

    # Load entire raw document content
    manual_old_text = load_document_content(os.path.join(DOCUMENT_LIBRARY_PATH, MANUAL_OLD_NAME))
    manual_newest_text = load_document_content(os.path.join(DOCUMENT_LIBRARY_PATH, MANUAL_NEWEST_NAME))

    if manual_old_text is None or manual_newest_text is None:
        print("Error: Could not load full text from one or more employee manuals. Please check paths and content. Exiting.")
        exit()

    print(f"Manual {MANUAL_OLD_NAME} loaded with {len(manual_old_text)} characters.")
    print(f"Manual {MANUAL_NEWEST_NAME} loaded with {len(manual_newest_text)} characters.")

    compiled_report_by_topic = {}
    print("\n--- Starting Compilation and Contradiction Detection ---")

    for topic in KEY_POLICY_AREAS:
        # Pass raw texts directly to the compilation function
        compiled_data = compile_and_compare_policy_area(topic, manual_old_text, manual_newest_text, llm)
        compiled_report_by_topic[topic] = compiled_data
        print(f"\n{'='*20} End of Analysis for '{topic}' {'='*20}\n")

    print("\n--- Full Compiled Report (Console Preview) ---")
    for topic, data in compiled_report_by_topic.items():
        print(f"\n### Policy Area: {topic}\n")
        print(data["llm_response"])
        print("\n--------------------------------------------------")

    output_directory = "./output_reports_version_10"
    os.makedirs(output_directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_filename = os.path.join(output_directory, f"report_{timestamp}.pdf")
    try:
        write_report_to_pdf(compiled_report_by_topic, pdf_output_filename)
        print(f"Successfully saved compiled report to PDF: {pdf_output_filename}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")

    print("\nProgram Finished.")
