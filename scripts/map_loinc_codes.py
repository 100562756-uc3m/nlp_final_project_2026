"""
LOINC Code Discovery & Dataset Audit Tool
-----------------------------------------
PURPOSE:
    This script performs an initial 'reconnaissance' of the DailyMed SPL dataset.
    Since drug labels are submitted by different manufacturers, they often use
    varying names for the same clinical section (e.g., 'Side Effects' vs 'Adverse Reactions').
    
    This script identifies every unique LOINC code (the medical standard code) in the 
    dataset and lists all the different human-readable titles associated with them.

OUTPUT:
    Generates 'data/loinc_audit_results.json' which acts as a guide for creating 
    the standardization dictionary (MASTER_LOINC_MAP).

USAGE:
    Place your raw DailyMed .zip in 'data/raw/' and run:
    python3 scripts/map_loinc_codes.py
"""

import os
import xml.etree.ElementTree as ET
import zipfile
import io
import json
from collections import defaultdict

# HL7 V3 Namespace: SPL XML files use this standard. 
# We need this to correctly find tags like <section> or <title>.
NS = {'v3': 'urn:hl7-org:v3'}

def audit_xml(xml_content, global_dict):
    """
    Parses an individual XML drug label to find section codes and titles.

    Args:
        xml_content (bytes): The raw XML content of a single drug label.
        global_dict (defaultdict): A shared dictionary where keys are LOINC codes 
                                   and values are sets of discovered titles.
    """
    try:
        # Convert raw bytes into an XML tree structure
        root = ET.fromstring(xml_content)
        
        # Find every <section> tag in the document (including nested sub-sections)
        for section in root.findall('.//v3:section', NS):
            # 1. Identify the LOINC Code (e.g., 34067-9)
            code_node = section.find('v3:code', NS)
            code = code_node.get('code') if code_node is not None else "None"
            
            # 2. Extract a human-readable name for this code
            # First, try the 'displayName' attribute (e.g., 'ADVERSE REACTIONS SECTION')
            display_name = None
            if code_node is not None:
                display_name = code_node.get('displayName')
            
            # If displayName is missing, fallback to the actual <title> text in the XML
            if not display_name:
                title_node = section.find('v3:title', NS)
                if title_node is not None and title_node.text:
                    display_name = title_node.text.strip()
            
            # 3. Clean up the text (remove excessive whitespace/newlines)
            # Default to 'UNTITLED' if no name or title exists
            name = " ".join((display_name or "UNTITLED").split())
            
            # 4. Add to our global tracker (sets automatically handle duplicates)
            global_dict[code].add(name)
    except Exception:
        # If an XML is malformed, skip it and move to the next
        pass

def run_dataset_audit(zip_path):
    """
    Handles the high-level logic of opening the DailyMed 'Zip-within-a-Zip' structure.

    Args:
        zip_path (str): Path to the main dm_spl_release_human_rx_part1.zip file.
    """
    # Keys: LOINC code string | Values: Set of unique titles found
    loinc_discovery = defaultdict(set)
    
    # Check if the file exists before attempting to open
    if not os.path.exists(zip_path):
        print(f"Error: Could not find {zip_path}")
        return

    # Open the primary Master Zip
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        # Get list of all internal product .zip files
        all_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
        print(f"Auditing {len(all_zips)} drug products... this may take several minutes.")

        for iz_name in all_zips:
            try:
                # Open the internal product zip (e.g., 2024_uuid.zip)
                with outer_zip.open(iz_name) as inner_zip_data:
                    # Treat the data as a file stream using io.BytesIO
                    with zipfile.ZipFile(io.BytesIO(inner_zip_data.read())) as inner:
                        # Find the actual .xml file inside the product zip
                        xml_name = [f for f in inner.namelist() if f.endswith('.xml')][0]
                        with inner.open(xml_name) as f:
                            # Pass the XML content to our audit function
                            audit_xml(f.read(), loinc_discovery)
            except Exception:
                # Skip any product folder that is empty or contains errors
                continue

    # 5. Format the results for saving
    # We convert 'set' objects to 'sorted list' so they are JSON-serializable
    final_dict = {code: sorted(list(names)) for code, names in loinc_discovery.items()}
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save the full audit report
    with open('data/loinc_audit_results.json', 'w') as f:
        json.dump(final_dict, f, indent=2)
        
    print(f"Audit complete!")
    print(f"Found {len(final_dict)} unique section codes.")
    print("Report saved to: data/loinc_audit_results.json")

if __name__ == "__main__":
    # Point this to the location of your downloaded DailyMed archive
    run_dataset_audit('data/raw/dm_spl_release_human_rx_part1.zip')