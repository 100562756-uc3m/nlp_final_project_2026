# NLP Final Project 2026 - DailyMed RAG Chatbot
**Master in Machine Learning for Health**

## Project Overview
This application implements a high-resolution Retrieval-Augmented Generation (RAG) system specialized in FDA drug labels (DailyMed). It utilizes semantic grouping and metadata injection to provide accurate, grounded answers from a clinical database of over 15,000 medications while strictly mitigating hallucinations.

## Functional Specifications
- **Vector Database**: FAISS index containing **~693,000 "Smart Chunks"** generated from official SPL XML files.
- **Embeddings**: `all-mpnet-base-v2` (768 dimensions) for superior medical semantic precision.
- **LLM**: Powered by **Llama 3.1 8B** via the UC3M API.
- **Multi-language**: Seamless support for English and Spanish queries with automated query translation and response mapping.
- **Citations**: Response includes inline citations linked to specific clinical sections (e.g., `SAFETY_RISK`, `PHARMACOLOGY`).

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/nlp_final_project_2026.git](https://github.com/your-username/nlp_final_project_2026.git)
   cd nlp_final_project_2026
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **External Data Setup**:
   Because the vector database and clinical datasets exceed GitHub's file size limits, the core data must be added manually:
   * Download the `data.zip` file from the provided [Google Drive link](https://drive.google.com/file/d/1Dz8N4xBRboWqaSixpDqQRUfnRCr-uKD6/view?usp=drive_link).
   * Uncompress the folder into the project root so you have a directory named `/data` containing the FAISS index and JSONL files.

4. **Configure Environment Variables**:
   Create a file named .env in the root directory and add your UC3M API key:
    ```bash
    UC3M_API_KEY=your_actual_key_here
    ```
    *(Note: The .env file is excluded from version control via .gitignore for security).*

5. **Run the Application**:
   ```bash
   python3 -m streamlit run app.py
   ```

## Design Decisions

### 1. Smart Semantic Chunking
Instead of simple character-count splitting, this system uses **Dynamic Accumulation** based on LOINC codes. Documents are parsed from XML and sections are grouped into 6 high-level **Semantic Super-Groups**:

* **Safety Risk:** Boxed warnings, contraindications, and precautions.
* **Clinical Usage:** Indications, dosages, and administration.
* **Adverse Interactions:** Side effects and drug interactions.
* **Special Populations:** Pregnancy, pediatric, and geriatric data.
* **Pharmacology:** Mechanism of action and pharmacokinetics.
* **Product Logistics:** Ingredients and supply information.

This ensures that related medical facts (like dosage for different age groups) stay within the same context window.

### 2. Contextual Metadata Injection
To prevent the LLM from confusing information between different drugs, each chunk is prepended with a header:
`DRUG: [Name] | GROUP: [Category] | PART: [X/Y]`

This **"Breadcrumb"** approach ensures the vector representation and the LLM generation are both anchored to the specific drug entity.

### 3. Cross-Language Retrieval Loop
To maximize accuracy, non-English queries undergo a three-step process:

1.  **Detection:** Identifying the source language.
2.  **Translation:** Converting the query to English to match the medical database.
3.  **Retrieval & Synthesis:** Searching the FAISS index and generating the English answer, then translating the final grounded response back to the user's original language.

---

## Project Structure

* **`app.py`**: Streamlit web interface and session management.
* **`src/system.py`**: Core logic for FAISS loading, retrieval, and translation loops.
* **`src/constants.py`**: LOINC mappings and Super-Group definitions.
* **`src/api.py`**: Connection logic for the UC3M Llama 3.1 API.
* **`src/prompts.py`**: System instructions for grounded medical answering.
* **`data/vector_db/`**: Directory containing the FAISS index and metadata.
