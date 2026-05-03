import os
import xml.etree.ElementTree as ET
import zipfile
import io
import json
import re
from src.constants import MASTER_LOINC_MAP

NS = {'v3': 'urn:hl7-org:v3'}

def get_clean_text(element):
    if element is None: return ""
    return " ".join("".join(element.itertext()).split())

def parse_section_recursive(section_element):
    """Recursively parses sections and sub-sections."""
    code_node = section_element.find('v3:code', NS)
    code = code_node.get('code') if code_node is not None else "None"
    
    title_node = section_element.find('v3:title', NS)
    title = get_clean_text(title_node) if title_node is not None else "Untitled"
    
    text_node = section_element.find('v3:text', NS)
    content = get_clean_text(text_node)
    
    # Standardize category
    category = MASTER_LOINC_MAP.get(code, "UNCLASSIFIED")

    return {
        "loinc": code,
        "category": category,
        "title": title,
        "content": content,
        "sub_sections": [parse_section_recursive(sub) for sub in section_element.findall('v3:component/v3:section', NS)]
    }

def run_conversion(zip_path, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        inner_zips = [f for f in outer_zip.namelist() if f.endswith('.zip')]
        print(f"Converting {len(inner_zips)} products...")

        for i, z_name in enumerate(inner_zips):
            try:
                with outer_zip.open(z_name) as z_data:
                    with zipfile.ZipFile(io.BytesIO(z_data.read())) as inner:
                        xml_name = [f for f in inner.namelist() if f.endswith('.xml')][0]
                        with inner.open(xml_name) as f:
                            root = ET.fromstring(f.read())
                            
                            drug_name = get_clean_text(root.find('.//v3:title', NS))
                            set_id = root.find('.//v3:setId', NS).get('root')
                            
                            body_sections = root.findall('.//v3:structuredBody/v3:component/v3:section', NS)
                            
                            result = {
                                "drug_name": drug_name,
                                "set_id": set_id,
                                "sections": [parse_section_recursive(s) for s in body_sections]
                            }
                            
                            with open(os.path.join(out_dir, f"{set_id}.json"), 'w', encoding='utf-8') as jf:
                                json.dump(result, jf, indent=2, ensure_ascii=False)
                if i % 500 == 0: print(f"Progress: {i}/{len(inner_zips)} files converted.")
            except: continue

if __name__ == "__main__":
    run_conversion("data/raw/dm_spl_release_human_rx_part1.zip", "data/processed")
