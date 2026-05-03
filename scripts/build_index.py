import os
import json
import time
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import SUPER_GROUP_MAP, MAX_TOKEN_LENGTH, CHUNK_OVERLAP_TOKENS

# Mute warnings
import transformers
transformers.logging.set_verbosity_error()

TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def create_smart_chunks(json_dir, jsonl_handle):
    documents = []
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

                flat_list = []
                def flatten(sections):
                    for s in sections:
                        flat_list.append(s)
                        if s.get('sub_sections'): flatten(s['sub_sections'])
                flatten(data.get('sections', []))

                buffer_text, buffer_tokens, current_super_group = [], 0, None

                for section in flat_list:
                    cat = section['category']
                    super_group = SUPER_GROUP_MAP.get(cat, "GENERAL_INFO")
                    section_content = f"## {section['title']}\n{section['content']}\n\n"
                    section_tokens = len(TOKENIZER.encode(section_content, add_special_tokens=False))

                    # Oversized sections
                    if section_tokens > MAX_TOKEN_LENGTH:
                        if buffer_text:
                            save_to_docs(documents, drug, set_id, current_super_group, buffer_text, jsonl_handle)
                            buffer_text, buffer_tokens = [], 0
                        sub_splits = backup_splitter.split_text(section_content)
                        for i, split in enumerate(sub_splits):
                            save_to_docs(documents, drug, set_id, super_group, [split], jsonl_handle, part=(i+1, len(sub_splits)))
                        continue

                    # Accumulation
                    if (current_super_group and super_group != current_super_group) or \
                       (buffer_tokens + section_tokens > MAX_TOKEN_LENGTH):
                        if buffer_text:
                            save_to_docs(documents, drug, set_id, current_super_group, buffer_text, jsonl_handle)
                        buffer_text, buffer_tokens = [], 0

                    current_super_group = super_group
                    buffer_text.append(section_content)
                    buffer_tokens += section_tokens

                if buffer_text:
                    save_to_docs(documents, drug, set_id, current_super_group, buffer_text, jsonl_handle)
        except Exception:
            continue
    return documents

def save_to_docs(documents, drug, set_id, group, text_list, jsonl_handle, part=None):
    content = "".join(text_list)
    seq_str = f" | PART: {part[0]}/{part[1]}" if part else " | PART: 1/1"
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
    
    # Save to JSONL Mirror (New Line per Chunk)
    json_record = {"content": page_content, "metadata": metadata}
    jsonl_handle.write(json.dumps(json_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    start_time = time.time()
    
    # --- PHASE 1: CHUNKING & LIVE JSONL SAVING ---
    mirror_path = "data/vector_db/smart_index_inspect.jsonl"
    os.makedirs(os.path.dirname(mirror_path), exist_ok=True)
    
    print(f"\n Starting chunking and saving to JSONL...")
    with open(mirror_path, 'w', encoding='utf-8') as jsonl_f:
        docs = create_smart_chunks('data/processed', jsonl_f)
    
    total_docs = len(docs)
    print(f" Chunking Complete. Generated {total_docs} chunks. Inspection file: {mirror_path}")

    # --- PHASE 2: VECTORIZATION ---
    print(f" Initializing MPNet Model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    batch_size = 500  
    checkpoint_every = 25000
    db = None
    vector_path = "data/vector_db/smart_index"

    print(f" Starting Vectorization (Target: {total_docs} chunks)...")
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
        
        current_count = i + len(batch)
        elapsed = (time.time() - start_time) / 60
        print(f" [{current_count}/{total_docs}] ({(current_count/total_docs)*100:.1f}%) | {elapsed:.1f} min elapsed")

        # Save FAISS checkpoint every 25k chunks
        if current_count % checkpoint_every == 0:
            print(f" Saving FAISS checkpoint at {current_count} chunks...")
            db.save_local(vector_path)

    # Final Save
    db.save_local(vector_path)
    print(f"\n Finished! Total chunks indexed: {total_docs}. Final index saved to {vector_path}")
