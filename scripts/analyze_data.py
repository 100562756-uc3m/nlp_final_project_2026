"""
This script performs a statistical analysis of processed medical drug labels in JSON format. 
Its primary purpose is to help developers understand the character-length distribution of 
different clinical sections (e.g., INDICATIONS, ADVERSE REACTIONS) across the dataset.

By calculating averages, 90th percentiles, and maximum lengths, the script estimates 
how many 'chunks' will be generated for the FAISS vector database. This allows for 
better tuning of retrieval parameters and helps prevent data loss due to context window 
overflow in the LLM.

HOW TO RUN:
Ensure you have processed the raw XML data into JSON first. From the project root, run:
python scripts/analyze_data.py
"""

import os
import json
from collections import defaultdict
import numpy as np
from src.constants import MAX_TOKEN_LENGTH

def analyze_section_lengths(json_dir):
    """
    Iterates through a directory of drug label JSON files to calculate content 
    length statistics per clinical category.

    Args:
        json_dir (str): Path to the directory containing processed .json files.

    Returns:
        int: The total number of estimated chunks for the entire dataset.
    """
    stats = defaultdict(list)
    total_files = 0
    
    print(f"Analyzing processed JSONs in {json_dir}...")
    
    # Identify all JSON files in the target directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for fn in json_files:
        total_files += 1
        with open(os.path.join(json_dir, fn), 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Temporary storage for current file's section lengths
            groups = defaultdict(int)

            def walk(sections):
                """Recursively traverses nested sections to capture all content."""
                for s in sections:
                    cat = s.get('standard_category', 'OTHER')
                    content_len = len(s.get('content', ""))
                    groups[cat] += content_len
                    # Handle nested sub-sections if they exist
                    if s.get('sub_sections'):
                        walk(s['sub_sections'])

            # Start the recursive walk from the root sections list
            walk(data.get('sections', []))
            
            # Record the aggregated lengths for this specific drug
            for cat, length in groups.items():
                stats[cat].append(length)

    # ─── REPORT GENERATION ───
    print(f"\n{'='*85}")
    print(f"DETAILED SECTION ANALYSIS REPORT ({total_files} Drugs)")
    print(f"{'='*85}")
    # Header defining the columns of the statistical report
    header = f"{'Category':<25} | {'Avg':<7} | {'90th%':<7} | {'Max':<8} | {'>'+str(MAX_TOKEN_LENGTH):<10} | {'Est. Chunks'}"
    print(header)
    print("-" * 85)
    
    total_estimated_chunks = 0
    
    # Sort categories by average length descending
    sorted_cats = sorted(stats.items(), key=lambda x: np.mean(x[1]), reverse=True)

    for cat, lengths in sorted_cats:
        avg = int(np.mean(lengths))
        p90 = int(np.percentile(lengths, 90))
        max_l = int(np.max(lengths))
        # Counts how many sections exceed the single-chunk token limit
        over_limit = sum(1 for l in lengths if l > MAX_TOKEN_LENGTH)
        
        # Estimate chunks based on the MAX_TOKEN_LENGTH setting
        # Uses ceiling division to ensure partial chunks are counted as full units
        cat_chunks = sum([max(1, int(np.ceil(l / MAX_TOKEN_LENGTH))) for l in lengths])
        total_estimated_chunks += cat_chunks
        
        print(f"{cat:<25} | {avg:<7} | {p90:<7} | {max_l:<8} | {over_limit:<10} | {cat_chunks}")

    print(f"{'-'*85}")
    print(f"{'TOTAL ESTIMATED CHUNKS FOR FAISS:':<68} {total_estimated_chunks}")
    print(f"{'='*85}")
    
    return total_estimated_chunks

if __name__ == "__main__":
    # Logic to ensure the directory is valid before processing
    json_path = 'data/processed'
    if os.path.exists(json_path):
        analyze_section_lengths(json_path)
    else:
        # Error handling if the prerequisite JSON files are missing
        print(f"Error: {json_path} not found. Run scripts/xml_to_json.py first.")