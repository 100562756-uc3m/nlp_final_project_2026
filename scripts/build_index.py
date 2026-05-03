"""
This script converts processed drug label JSON files into a searchable vector 
database (FAISS). It performs "Smart Chunking" by grouping related clinical 
sections together, transforms the text into high-dimensional numerical vectors 
using the 'all-mpnet-base-v2' model, and saves both a searchable index and an 
inspection JSONL file.

WORKFLOW:
1. PHASE 1: Parses JSON files, flattens nested sections, and groups them 
   by 'Super Groups' (defined in constants.py).
2. PHASE 2: Generates text embeddings in batches to optimize memory usage.
3. PHASE 3: Saves a local FAISS index for the RAG system and a JSONL mirror 
   for manual inspection of the indexed content.

HOW TO RUN:
1. Ensure 'data/processed/' contains the JSON files from the extraction step.
2. Run from the project root using:
   python scripts/build_index.py
================================================================================
"""
import os
import json
import time
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import SUPER_GROUP_MAP, MAX_TOKEN_LENGTH, CHUNK_OVERLAP_TOKENS

# Mute standard transformer warnings to keep the console output clean
import transformers
transformers.logging.set_verbosity_error()

# Initialize the tokenizer globally to calculate accurate token counts for chunking
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def create_smart_chunks(json_dir, jsonl_handle):
    """
    Processes drug JSON files into 'smart chunks'. Instead of cutting text 
    randomly, it attempts to keep related clinical sections (like Dosage and 
    Administration) together until the token limit is reached.

    Args:
        json_dir (str): Path to directory containing processed JSON files.
        jsonl_handle (file_object): Open file handle for writing the mirror JSONL.

    Returns:
        list[Document]: A list of LangChain Document objects ready for embedding.
    """
    documents = []

    # Recursive splitter used only as a fallback for extremely long sections
    backup_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(MAX_TOKEN_LENGTH * 3.2), 
        chunk_overlap=int(CHUNK_OVERLAP_TOKENS * 3.2),
        separators=["\n\n", "\n", ". ", " "]
    )
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    total_files = len(json_files)
    print(f" Starting chunking for {total_files} drugs...")

    for idx, fn in enumerate(json_files):
        try:
            with open(os.path.join(json_dir, fn), 'r', encoding='utf-8') as f:
                data = json.load(f)
                drug = data.get('drug_name', 'Unknown')
                set_id = data.get('set_id', 'Unknown')
                
                if idx % 500 == 0:
                    print(f" [File {idx}/{total_files}] Processing: {drug[:30]}...")

                # Recursively flatten nested sub-sections into a linear list for processing
                flat_list = []
                def flatten(sections):
                    for s in sections:
                        flat_list.append(s)
                        if s.get('sub_sections'): flatten(s['sub_sections'])
                flatten(data.get('sections', []))

                buffer_text, buffer_tokens, current_super_group = [], 0, None

                for section in flat_list:
                    cat = section['category']
                    # Map the raw LOINC category to a broader 'Super Group'
                    super_group = SUPER_GROUP_MAP.get(cat, "GENERAL_INFO")
                    section_content = f"## {section['title']}\n{section['content']}\n\n"
                    section_tokens = len(TOKENIZER.encode(section_content, add_special_tokens=False))

                    # SCENARIO A: Single section is already bigger than the allowed context window
                    # We must split this specific section into parts
                    if section_tokens > MAX_TOKEN_LENGTH:
                        if buffer_text:
                            save_to_docs(documents, drug, set_id, current_super_group, buffer_text, jsonl_handle)
                            buffer_text, buffer_tokens = [], 0
                        sub_splits = backup_splitter.split_text(section_content)
                        for i, split in enumerate(sub_splits):
                            save_to_docs(documents, drug, set_id, super_group, [split], jsonl_handle, part=(i+1, len(sub_splits)))
                        continue

                    # SCENARIO B: The section fits, but adding it exceeds the chunk limit 
                    # OR the clinical category has changed (e.g., moving from Dosage to Warnings)
                    if (current_super_group and super_group != current_super_group) or \
                       (buffer_tokens + section_tokens > MAX_TOKEN_LENGTH):
                        if buffer_text:
                            save_to_docs(documents, drug, set_id, current_super_group, buffer_text, jsonl_handle)
                        buffer_text, buffer_tokens = [], 0

                    # SCENARIO C: Accumulate current section into the buffer
                    current_super_group = super_group
                    buffer_text.append(section_content)
                    buffer_tokens += section_tokens

                # Clean up any remaining text in the buffer after finishing the file
                if buffer_text:
                    save_to_docs(documents, drug, set_id, current_super_group, buffer_text, jsonl_handle)
        except Exception:
            continue # Skip files with corrupted JSON structure
    return documents

def save_to_docs(documents, drug, set_id, group, text_list, jsonl_handle, part=None):
    """
    Finalizes a chunk by adding metadata headers and saving it to two places:
    1. The 'documents' list for FAISS vectorization.
    2. The 'smart_index_inspect.jsonl' file for human debugging.

    Args:
        documents (list): Target list for Document objects.
        drug (str): Name of the medication.
        set_id (str): Unique FDA identifier for the drug label.
        group (str): Super Group name.
        text_list (list): List of text strings to be joined into a chunk.
        jsonl_handle (file): File handle for writing metadata.
        part (tuple): Optional (current_part, total_parts) for segmented chunks.
    """
    content = "".join(text_list)
    seq_str = f" | PART: {part[0]}/{part[1]}" if part else " | PART: 1/1"
    # Inject drug and group headers into the text so the embedding model "knows" 
    # the context of the chunk
    header = f"DRUG: {drug} | GROUP: {group}{seq_str}\n"
    
    page_content = header + content
    metadata = {
        "drug": drug, 
        "set_id": set_id,
        "group": group, 
        "part": part[0] if part else 1,
        "total_parts": part[1] if part else 1
    }
    
    # Add to FAISS list
    documents.append(Document(page_content=page_content, metadata=metadata))
    
    # Mirror the entry into a JSONL file for easy inspection later
    json_record = {"content": page_content, "metadata": metadata}
    jsonl_handle.write(json.dumps(json_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    start_time = time.time()
    
    ## --- PHASE 1: CHUNKING & LIVE JSONL SAVING ---
    mirror_path = "data/vector_db/smart_index_inspect.jsonl"
    os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
    
    print(f"\n Starting chunking and saving to JSONL...")
    with open(mirror_path, 'w', encoding='utf-8') as jsonl_f:
        docs = create_smart_chunks('data/processed', jsonl_f)
    
    total_docs = len(docs)
    print(f" Chunking Complete. Generated {total_docs} chunks. Inspection file: {mirror_path}")

    # --- PHASE 2: VECTORIZATION ---
    print(f" Initializing MPNet Model...")
    # Uses HuggingFace embeddings based on the 'all-mpnet-base-v2' model
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    batch_size = 500  # Number of documents to vectorize at once (saves memory)
    checkpoint_every = 25000 # Save intermediate progress to disk
    db = None
    vector_path = "data/vector_db/smart_index"

    print(f" Starting Vectorization (Target: {total_docs} chunks)...")
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        if db is None:
            # Initialize FAISS with the first batch
            db = FAISS.from_documents(batch, embeddings)
        else:
            # Add subsequent batches to the existing index
            db.add_documents(batch)
        
        current_count = i + len(batch)
        elapsed = (time.time() - start_time) / 60
        print(f" [{current_count}/{total_docs}] ({(current_count/total_docs)*100:.1f}%) | {elapsed:.1f} min elapsed")

        # Save FAISS checkpoint periodically to avoid losing progress on large datasets
        if current_count % checkpoint_every == 0:
            print(f" Saving FAISS checkpoint at {current_count} chunks...")
            db.save_local(vector_path)

    # Final Save of the completed vector database
    db.save_local(vector_path)
    print(f"\n Finished! Total chunks indexed: {total_docs}. Final index saved to {vector_path}")
