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

# import os
# import json
# import time
# from transformers import AutoTokenizer
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from src.constants import SUPER_GROUP_MAP, MAX_TOKEN_LENGTH, CHUNK_OVERLAP_TOKENS

# # Mute the tokenizer warning
# import transformers
# transformers.logging.set_verbosity_error()

# TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# def create_smart_chunks(json_dir):
#     documents = []
#     backup_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=int(MAX_TOKEN_LENGTH * 3.2), 
#         chunk_overlap=int(CHUNK_OVERLAP_TOKENS * 3.2),
#         separators=["\n\n", "\n", ". ", " "]
#     )
    
#     json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
#     total_files = len(json_files)
#     print(f" Starting chunking for {total_files} drugs...")

#     for idx, fn in enumerate(json_files):
#         try:
#             with open(os.path.join(json_dir, fn), 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 drug = data.get('drug_name', 'Unknown')
                
#                 if idx % 500 == 0:
#                     print(f" [File {idx}/{total_files}] Current Drug: {drug[:30]}...")

#                 flat_list = []
#                 def flatten(sections):
#                     for s in sections:
#                         flat_list.append(s)
#                         if s.get('sub_sections'): flatten(s['sub_sections'])
#                 flatten(data.get('sections', []))

#                 buffer_text, buffer_tokens, current_super_group = [], 0, None

#                 for section in flat_list:
#                     cat = section['category']
#                     super_group = SUPER_GROUP_MAP.get(cat, "GENERAL_INFO")
#                     section_content = f"## {section['title']}\n{section['content']}\n\n"
#                     section_tokens = len(TOKENIZER.encode(section_content, add_special_tokens=False))

#                     if section_tokens > MAX_TOKEN_LENGTH:
#                         if buffer_text:
#                             save_to_docs(documents, drug, current_super_group, buffer_text)
#                             buffer_text, buffer_tokens = [], 0
#                         sub_splits = backup_splitter.split_text(section_content)
#                         for i, split in enumerate(sub_splits):
#                             save_to_docs(documents, drug, super_group, [split], part=(i+1, len(sub_splits)))
#                         continue

#                     if (current_super_group and super_group != current_super_group) or \
#                        (buffer_tokens + section_tokens > MAX_TOKEN_LENGTH):
#                         if buffer_text:
#                             save_to_docs(documents, drug, current_super_group, buffer_text)
#                         buffer_text, buffer_tokens = [], 0

#                     current_super_group = super_group
#                     buffer_text.append(section_content)
#                     buffer_tokens += section_tokens

#                 if buffer_text:
#                     save_to_docs(documents, drug, current_super_group, buffer_text)
#         except Exception as e:
#             continue
#     return documents

# def save_to_docs(documents, drug, group, text_list, part=None):
#     content = "".join(text_list)
#     seq_str = f" | PART: {part[0]}/{part[1]}" if part else " | PART: 1/1"
#     header = f"DRUG: {drug} | GROUP: {group}{seq_str}\n"
#     documents.append(Document(
#         page_content=header + content,
#         metadata={"drug": drug, "group": group, "part": part[0] if part else 1}
#     ))

# if __name__ == "__main__":
#     # --- PHASE 1: CHUNKING ---
#     start_time = time.time()
#     docs = create_smart_chunks('data/processed')
#     total_docs = len(docs)
#     print(f"\n Chunking Complete. Generated {total_docs} smart chunks.")
    
#     # --- PHASE 2: EMBEDDING ---
#     print(f" Initializing MPNet Model...")
#     embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
#     # We will save a 'checkpoint' every 25,000 chunks to avoid losing data
#     batch_size = 5000
#     checkpoint_every = 25000 
#     db = None
    
#     vector_path = "data/vector_db/smart_index"
#     os.makedirs(os.path.dirname(vector_path), exist_ok=True)

#     print(f" Starting Vectorization (Target: {total_docs} chunks)...")
    
#     for i in range(0, total_docs, batch_size):
#         batch = docs[i : i + batch_size]
        
#         if db is None:
#             db = FAISS.from_documents(batch, embeddings)
#         else:
#             db.add_documents(batch)
        
#         # Evolution printing
#         current_count = i + len(batch)
#         elapsed = time.time() - start_time
#         print(f" [{current_count}/{total_docs}] ({ (current_count/total_docs)*100:.1f}%) | Elapsed: {elapsed/60:.1f} min")

#         # Checkpointing: Save to disk every 25k chunks
#         if current_count % checkpoint_every == 0:
#             print(f" Saving checkpoint to disk...")
#             db.save_local(vector_path)

#     # --- FINAL SAVE ---
#     db.save_local(vector_path)
#     print(f"\n FINISHED! Total time: {(time.time() - start_time)/3600:.2f} hours.")
#     print(f"Final index saved to {vector_path}")

# # import os
# # import json
# # import logging
# # from transformers import AutoTokenizer
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_core.documents import Document
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from src.constants import SUPER_GROUP_MAP, MAX_TOKEN_LENGTH, CHUNK_OVERLAP_TOKENS

# # # Mute the tokenizer warning
# # import transformers
# # transformers.logging.set_verbosity_error()

# # # Initialize Tokenizer
# # TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# # def create_smart_chunks(json_dir):
# #     documents = []
    
# #     # We reduce the character multiplier slightly (from 4 to 3.2) 
# #     # to be safer against dense text/tables.
# #     backup_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=int(MAX_TOKEN_LENGTH * 3.2), 
# #         chunk_overlap=int(CHUNK_OVERLAP_TOKENS * 3.2),
# #         separators=["\n\n", "\n", ". ", " "]
# #     )
    
# #     json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
# #     print(f"Processing {len(json_files)} drugs with Hybrid Overlap Strategy...")

# #     for fn in json_files:
# #         try:
# #             with open(os.path.join(json_dir, fn), 'r', encoding='utf-8') as f:
# #                 data = json.load(f)
# #                 drug = data.get('drug_name', 'Unknown')
                
# #                 flat_list = []
# #                 def flatten(sections):
# #                     for s in sections:
# #                         flat_list.append(s)
# #                         if s.get('sub_sections'): flatten(s['sub_sections'])
# #                 flatten(data.get('sections', []))

# #                 buffer_text = []
# #                 buffer_tokens = 0
# #                 current_super_group = None

# #                 for section in flat_list:
# #                     cat = section['category']
# #                     super_group = SUPER_GROUP_MAP.get(cat, "GENERAL_INFO")
                    
# #                     section_content = f"## {section['title']}\n{section['content']}\n\n"
                    
# #                     # Truncation=False avoids the warning during length check
# #                     section_tokens = len(TOKENIZER.encode(section_content, add_special_tokens=False))

# #                     # --- STEP 1: HANDLE OVERSIZED SINGLE SECTIONS ---
# #                     if section_tokens > MAX_TOKEN_LENGTH:
# #                         if buffer_text:
# #                             save_to_docs(documents, drug, current_super_group, buffer_text)
# #                             buffer_text, buffer_tokens = [], 0
                        
# #                         sub_splits = backup_splitter.split_text(section_content)
# #                         for i, split in enumerate(sub_splits):
# #                             save_to_docs(documents, drug, super_group, [split], part=(i+1, len(sub_splits)))
# #                         continue

# #                     # --- STEP 2: ACCUMULATE SMALL RELATED SECTIONS ---
# #                     if (current_super_group and super_group != current_super_group) or \
# #                        (buffer_tokens + section_tokens > MAX_TOKEN_LENGTH):
                        
# #                         if buffer_text:
# #                             save_to_docs(documents, drug, current_super_group, buffer_text)
                        
# #                         buffer_text = []
# #                         buffer_tokens = 0

# #                     current_super_group = super_group
# #                     buffer_text.append(section_content)
# #                     buffer_tokens += section_tokens

# #                 if buffer_text:
# #                     save_to_docs(documents, drug, current_super_group, buffer_text)
# #         except Exception as e:
# #             print(f"Error processing {fn}: {e}")
# #             continue

# #     return documents

# # def save_to_docs(documents, drug, group, text_list, part=None):
# #     content = "".join(text_list)
# #     seq_str = f" | PART: {part[0]}/{part[1]}" if part else " | PART: 1/1"
# #     header = f"DRUG: {drug} | GROUP: {group}{seq_str}\n"
    
# #     documents.append(Document(
# #         page_content=header + content,
# #         metadata={
# #             "drug": drug, 
# #             "group": group,
# #             "part": part[0] if part else 1
# #         }
# #     ))

# # if __name__ == "__main__":
# #     docs = create_smart_chunks('data/processed')
# #     print(f"Generated {len(docs)} smart chunks.")
    
# #     embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
# #     batch_size = 5000
# #     db = None
    
# #     for i in range(0, len(docs), batch_size):
# #         batch = docs[i : i + batch_size]
# #         if db is None:
# #             db = FAISS.from_documents(batch, embeddings)
# #         else:
# #             db.add_documents(batch)
# #         print(f"Indexed {min(i + batch_size, len(docs))}/{len(docs)} chunks...")

# #     db.save_local("data/vector_db/smart_index")
# #     print("Process complete. FAISS database is ready.")

# # import os
# # import json
# # from collections import defaultdict
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.vectorstores import FAISS
# # from langchain_core.documents import Document

# # def create_grouped_chunks(json_dir):
# #     documents = []
    
# #     for fn in os.listdir(json_dir):
# #         if not fn.endswith('.json'): continue
# #         with open(os.path.join(json_dir, fn), 'r', encoding='utf-8') as f:
# #             data = json.load(f)
# #             drug = data.get('drug_name', 'Unknown')
# #             set_id = data.get('set_id', 'Unknown')
            
# #             # Use a dict to group text by Master Category
# #             # Key: Category Name | Value: List of strings
# #             groups = defaultdict(list)

# #             def walk(sections):
# #                 for s in sections:
# #                     cat = s['standard_category']
# #                     if s.get('content'):
# #                         # Add the sub-heading to the text so we don't lose detail
# #                         group_text = f"--- {s['original_title']} ---\n{s['content']}"
# #                         groups[cat].append(group_text)
# #                     if s.get('sub_sections'):
# #                         walk(s['sub_sections'])

# #             walk(data.get('sections', []))

# #             # Now create one "Chunk" per Category for this drug
# #             for category, content_list in groups.items():
# #                 full_category_text = "\n\n".join(content_list)
                
# #                 # If a category is extremely long, we might still want to split it
# #                 # but for now, we keep the clinical block together.
# #                 text = f"DRUG: {drug}\nCATEGORY: {category}\nCONTENT:\n{full_category_text}"
                
# #                 documents.append(Document(
# #                     page_content=text,
# #                     metadata={
# #                         "drug": drug,
# #                         "category": category,
# #                         "set_id": set_id
# #                     }
# #                 ))
                
# #     return documents

# # if __name__ == "__main__":
# #     print("Loading JSON files and grouping sections...")
# #     docs = create_grouped_chunks('data/processed')
    
# #     print(f"Total grouped chunks to embed: {len(docs)}")
# #     # This number should be significantly lower and more meaningful!

# #     # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# #     embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
# #     batch_size = 2000
# #     vector_db = None

# #     print(f"Starting batch embedding...")
# #     for i in range(0, len(docs), batch_size):
# #         batch = docs[i : i + batch_size]
# #         if vector_db is None:
# #             vector_db = FAISS.from_documents(batch, embeddings)
# #         else:
# #             vector_db.add_documents(batch)
# #         print(f"Progress: {i + len(batch)}/{len(docs)} chunks indexed...")

# #     os.makedirs("data/vector_db", exist_ok=True)
# #     vector_db.save_local("data/vector_db/structured_drug_index")
# #     print("Grouped FAISS Index saved successfully!")