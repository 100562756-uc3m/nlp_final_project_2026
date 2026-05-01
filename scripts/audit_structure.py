import os
import xml.etree.ElementTree as ET
import zipfile
import io
import random
from transformers import AutoTokenizer

# Load Tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
NS = {'v3': 'urn:hl7-org:v3'}

def get_clean_text(element):
    if element is None: return ""
    return " ".join("".join(element.itertext()).split())

def analyze_xml_structure(zip_path, sample_size=20):
    results = []
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return []

    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        inner_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
        sample_zips = random.sample(inner_zips, min(sample_size, len(inner_zips)))
        
        for z_name in sample_zips:
            try:
                with outer_zip.open(z_name) as z_data:
                    with zipfile.ZipFile(io.BytesIO(z_data.read())) as inner:
                        xml_name = [f for f in inner.namelist() if f.endswith('.xml')][0]
                        with inner.open(xml_name) as f:
                            tree = ET.parse(f)
                            root = tree.getroot()
                            
                            title_node = root.find('.//v3:title', NS)
                            drug_name = get_clean_text(title_node) if title_node is not None else "Unknown"
                            
                            # Create a map of parent elements for looking "up"
                            parent_map = {c: p for p in root.iter() for c in p}
                            
                            sequence = []
                            # Find all sections
                            for section in root.findall('.//v3:section', NS):
                                code_node = section.find('v3:code', NS)
                                code = code_node.get('code') if code_node is not None else "NO_CODE"
                                
                                # Token count
                                text_node = section.find('v3:text', NS)
                                content = get_clean_text(text_node)
                                token_count = len(TOKENIZER.encode(content))
                                
                                # Safe Parent Detection
                                parent_code = "ROOT"
                                component = parent_map.get(section)
                                if component is not None:
                                    grandparent = parent_map.get(component)
                                    # If the grandparent is also a section, this is a sub-section
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

if __name__ == "__main__":
    MASTER_ZIP = "data/raw/dm_spl_release_human_rx_part1.zip"
    audit_data = analyze_xml_structure(MASTER_ZIP)
    
    if not audit_data:
        print("No data analyzed.")
    else:
        print(f"{'Order':<5} | {'Parent Code':<12} | {'Section Code':<12} | {'Tokens'}")
        print("-" * 60)
        for entry in audit_data[:2]: 
            print(f"\nDRUG: {entry['drug'][:50]}...")
            for i, s in enumerate(entry['sequence']):
                print(f"{i:<5} | {s['parent']:<12} | {s['code']:<12} | {s['tokens']}")