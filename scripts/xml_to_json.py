"""
This script extracts and processes Structured Product Labeling (SPL) data 
from DailyMed raw archives. It navigates a nested ZIP structure, parses 
medical XMLs, and normalizes clinical categories using LOINC codes.

The output is a collection of JSON files, where each file represents a 
single drug product with its clinical sections (indications, dosage, etc.) 
structured for indexing.

HOW TO RUN:
    1. Ensure the raw DailyMed ZIP is in 'data/raw/'.
    2. Run from the project root using:
       python scripts/xml_to_json.py
================================================================================
"""
import os
import xml.etree.ElementTree as ET
import zipfile
import io
import json
import re
from src.constants import MASTER_LOINC_MAP

# XML Namespace for HL7 V3 Structured Product Labeling (SPL)
NS = {'v3': 'urn:hl7-org:v3'}

def get_clean_text(element):
    """
    Extracts all nested text from an XML element, removes HTML-style tags, 
    and collapses multiple spaces/newlines into a single clean string.

    Args:
        element (ET.Element): The XML node to extract text from.
    Returns:
        str: Cleaned, plain-text string.
    """
    if element is None: return ""
    return " ".join("".join(element.itertext()).split())

def parse_section_recursive(section_element):
    """
    Recursively traverses clinical sections. This is necessary because SPL 
    documents often nest specific data (e.g., 'Dosage for Adults') as 
    sub-sections under broader headings (e.g., 'Dosage and Administration').

    Args:
        section_element (ET.Element): The XML section node to parse.
    Returns:
        dict: A structured dictionary containing text and nested sub-sections.
    """
    # Extract LOINC code for clinical classification
    code_node = section_element.find('v3:code', NS)
    code = code_node.get('code') if code_node is not None else "None"
    
    # Extract Section Header
    title_node = section_element.find('v3:title', NS)
    title = get_clean_text(title_node) if title_node is not None else "Untitled"
    
    # Extract Main Content
    text_node = section_element.find('v3:text', NS)
    content = get_clean_text(text_node)
    
    # Map LOINC code to a human-readable category (e.g., '34067-9' -> 'INDICATIONS')
    category = MASTER_LOINC_MAP.get(code, "UNCLASSIFIED")

    return {
        "loinc": code,
        "category": category,
        "title": title,
        "content": content,
        # Recursively find and parse sub-sections
        "sub_sections": [parse_section_recursive(sub) for sub in section_element.findall('v3:component/v3:section', NS)]
    }

def run_conversion(zip_path, out_dir):
    """
    Manages the high-level orchestration of the conversion process.
    It handles the "ZIP-within-a-ZIP" extraction, identifies the main XML 
    document for each drug, and saves the final processed data.

    Args:
        zip_path (str): Path to the master DailyMed release ZIP file.
        out_dir (str): Directory where processed JSON files will be saved.
    """
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # Open the master DailyMed archive
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        # Get list of individual drug ZIPs inside the master archive
        inner_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
        print(f"Converting {len(inner_zips)} products...")

        for i, z_name in enumerate(inner_zips):
            try:
                # Open the inner drug-specific ZIP
                with outer_zip.open(z_name) as z_data:
                    with zipfile.ZipFile(io.BytesIO(z_data.read())) as inner:
                        # Find the main medical XML label file
                        xml_name = [f for f in inner.namelist() if f.endswith('.xml')][0]
                        with inner.open(xml_name) as f:
                            root = ET.fromstring(f.read())
                            
                            # Extract Global Metadata: Drug Name and Unique Set ID
                            drug_name = get_clean_text(root.find('.//v3:title', NS))
                            set_id = root.find('.//v3:setId', NS).get('root')
                            
                            # Locate the primary clinical body sections
                            body_sections = root.findall('.//v3:structuredBody/v3:component/v3:section', NS)
                            
                            # Construct the final clinical object
                            result = {
                                "drug_name": drug_name,
                                "set_id": set_id,
                                "sections": [parse_section_recursive(s) for s in body_sections]
                            }
                            
                            # Save to disk as [set_id].json
                            with open(os.path.join(out_dir, f"{set_id}.json"), 'w', encoding='utf-8') as jf:
                                json.dump(result, jf, indent=2, ensure_ascii=False)
                
                # Report progress every 500 files
                if i % 500 == 0: print(f"Progress: {i}/{len(inner_zips)} files converted.")
            except: 
                # Skip corrupted archives or invalid XML structures
                continue

if __name__ == "__main__":
    # Define source and destination paths
    RAW_ZIP = "data/raw/dm_spl_release_human_rx_part1.zip"
    PROCESSED_DIR = "data/processed"
    
    run_conversion(RAW_ZIP, PROCESSED_DIR)