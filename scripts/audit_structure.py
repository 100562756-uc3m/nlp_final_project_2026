"""
This script analyzes the internal hierarchy of DailyMed SPL (Structured Product 
Labeling) files. It extracts drug labels from a nested ZIP structure, parses 
their XML content, and audits the relationship between parent sections and 
sub-sections. Additionally, it calculates the token count for each section to 
estimate the data size for RAG (Retrieval-Augmented Generation) pipelines.

DEPENDENCIES:
- transformers (AutoTokenizer)
- standard python libraries (os, xml, zipfile, io, random)

HOW TO RUN:
1. Ensure you have the raw data zip at 'data/raw/dm_spl_release_human_rx_part1.zip'.
2. Run from the project root using:
   python scripts/audit_structure.py
================================================================================
"""

import os
import xml.etree.ElementTree as ET
import zipfile
import io
import random
from transformers import AutoTokenizer

# --- CONFIGURATION & GLOBALS ---

# Load the specific tokenizer used by the embedding model (all-mpnet-base-v2)
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# XML Namespace for SPL documents (urn:hl7-org:v3)
NS = {'v3': 'urn:hl7-org:v3'}

# --- FUNCTIONS ---

def get_clean_text(element):
    """
    Extracts all text from an XML element and its children, removes extra 
    whitespace, and returns a single clean string.
    
    Args:
        element (ET.Element): The XML node to extract text from.
    Returns:
        str: Cleaned text string.
    """
    if element is None: return ""
    return " ".join("".join(element.itertext()).split())

def analyze_xml_structure(zip_path, sample_size=20):
    """
    The core logic of the audit. It opens a master ZIP, randomly samples 
    internal drug-specific ZIPs, extracts the XML, and maps the hierarchy.
    
    It specifically detects if a section is a top-level component or a 
    nested sub-section by looking at its "grandparent" XML tag.

    Args:
        zip_path (str): Path to the master DailyMed .zip file.
        sample_size (int): Number of random drugs to audit.
    Returns:
        list: A structured list of dictionaries containing drug names and 
              section sequences (code, parent_code, token_count).
    """
    results = []
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return []

    # Open the master ZIP file
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        # Identify all internal compressed files (each represents one drug)
        inner_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
        sample_zips = random.sample(inner_zips, min(sample_size, len(inner_zips)))
        
        for z_name in sample_zips:
            try:
                # Read the inner ZIP into memory using io.BytesIO
                with outer_zip.open(z_name) as z_data:
                    with zipfile.ZipFile(io.BytesIO(z_data.read())) as inner:
                        # Find and parse the main XML document for the drug
                        xml_name = [f for f in inner.namelist() if f.endswith('.xml')][0]
                        with inner.open(xml_name) as f:
                            tree = ET.parse(f)
                            root = tree.getroot()
                            
                            # Extract Drug Name from the document title
                            title_node = root.find('.//v3:title', NS)
                            drug_name = get_clean_text(title_node) if title_node is not None else "Unknown"
                            
                            # Create a map of parent elements to allow upward navigation in the tree
                            parent_map = {c: p for p in root.iter() for c in p}
                            
                            sequence = []
                            # Find all 'section' tags within the document
                            for section in root.findall('.//v3:section', NS):
                                # Extract the LOINC/Section Code (e.g., 34067-9 for Indications)
                                code_node = section.find('v3:code', NS)
                                code = code_node.get('code') if code_node is not None else "NO_CODE"
                                
                                # Process the text to calculate token length
                                text_node = section.find('v3:text', NS)
                                content = get_clean_text(text_node)
                                token_count = len(TOKENIZER.encode(content))
                                
                                # --- PARENT DETECTION LOGIC ---
                                # We determine if this section is nested inside another section.
                                parent_code = "ROOT"
                                component = parent_map.get(section)
                                if component is not None:
                                    grandparent = parent_map.get(component)
                                    # If the grandparent is a section, this current node is a sub-section
                                    if grandparent is not None and grandparent.tag.endswith('section'):
                                        p_code_node = grandparent.find('v3:code', NS)
                                        parent_code = p_code_node.get('code') if p_code_node is not None else "PARENT_NO_CODE"
                                
                                sequence.append({
                                    "code": code,
                                    "parent": parent_code,
                                    "tokens": token_count
                                })
                            
                            results.append({"drug": drug_name, "sequence": sequence})
            except Exception as e:
                print(f"Skipping {z_name} due to error: {e}")
                continue
                
    return results

# --- EXECUTION ---

if __name__ == "__main__":
    # Define the data source
    MASTER_ZIP = "data/raw/dm_spl_release_human_rx_part1.zip"

    # Run the audit on 20 random drugs
    audit_data = analyze_xml_structure(MASTER_ZIP)
    
    if not audit_data:
        print("No data analyzed.")
    else:
        # Print a formatted table of the findings
        print(f"{'Order':<5} | {'Parent Code':<12} | {'Section Code':<12} | {'Tokens'}")
        print("-" * 60)
        # Display the first 2 drugs as a sample of the audit
        for entry in audit_data[:2]: 
            print(f"\nDRUG: {entry['drug'][:50]}...")
            for i, s in enumerate(entry['sequence']):
                print(f"{i:<5} | {s['parent']:<12} | {s['code']:<12} | {s['tokens']}")