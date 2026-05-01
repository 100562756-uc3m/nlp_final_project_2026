import os
import json
from collections import defaultdict
import numpy as np
from src.constants import CHUNK_SIZE # Importing your 1800 limit

def analyze_section_lengths(json_dir):
    stats = defaultdict(list)
    total_files = 0
    
    print(f"Analyzing processed JSONs in {json_dir}...")
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for fn in json_files:
        total_files += 1
        with open(os.path.join(json_dir, fn), 'r', encoding='utf-8') as f:
            data = json.load(f)
            groups = defaultdict(int)

            def walk(sections):
                for s in sections:
                    cat = s.get('standard_category', 'OTHER')
                    content_len = len(s.get('content', ""))
                    groups[cat] += content_len
                    if s.get('sub_sections'):
                        walk(s['sub_sections'])

            walk(data.get('sections', []))
            
            for cat, length in groups.items():
                stats[cat].append(length)

    print(f"\n{'='*85}")
    print(f"DETAILED SECTION ANALYSIS REPORT ({total_files} Drugs)")
    print(f"{'='*85}")
    header = f"{'Category':<25} | {'Avg':<7} | {'90th%':<7} | {'Max':<8} | {'>'+str(CHUNK_SIZE):<10} | {'Est. Chunks'}"
    print(header)
    print("-" * 85)
    
    total_estimated_chunks = 0
    
    # Sort categories by average length descending
    sorted_cats = sorted(stats.items(), key=lambda x: np.mean(x[1]), reverse=True)

    for cat, lengths in sorted_cats:
        avg = int(np.mean(lengths))
        p90 = int(np.percentile(lengths, 90))
        max_l = int(np.max(lengths))
        over_limit = sum(1 for l in lengths if l > CHUNK_SIZE)
        
        # Calculate chunks: if len=5000 and chunk=1800, that's ceil(5000/1800) = 3 chunks
        cat_chunks = sum([max(1, int(np.ceil(l / CHUNK_SIZE))) for l in lengths])
        total_estimated_chunks += cat_chunks
        
        print(f"{cat:<25} | {avg:<7} | {p90:<7} | {max_l:<8} | {over_limit:<10} | {cat_chunks}")

    print(f"{'-'*85}")
    print(f"{'TOTAL ESTIMATED CHUNKS FOR FAISS:':<68} {total_estimated_chunks}")
    print(f"{'='*85}")
    
    return total_estimated_chunks

if __name__ == "__main__":
    # Ensure this directory exists
    json_path = 'data/processed'
    if os.path.exists(json_path):
        analyze_section_lengths(json_path)
    else:
        print(f"Error: {json_path} not found. Run scripts/xml_to_json.py first.")