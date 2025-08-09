#!/usr/bin/env python3
"""
Script 2: Enrich with BioSample Metadata
Retrieves BioSample metadata and merges donor information
including sex, tissue, age, and collection details.

Author: Project 33 - Bias in Reference Genomes
Date: June 2025
"""

import time
import pandas as pd
import numpy as np
from Bio import Entrez
from xml.etree import ElementTree as ET
import re
from datetime import datetime
import requests
from collections import defaultdict

# Set your email for NCBI Entrez
Entrez.email = "research.project@example.edu"  # Research project email

# Known sex mappings from literature and databases
KNOWN_SEX_MAPPINGS = {
    # Human reference genomes
    'GRCh38': 'Female',  # Based on HuRef individual
    'CHM13': 'Male',     # Complete Hydatidiform Mole (paternal only)
    'GRCh37': 'Female',  # Same as GRCh38 base
    
    # Mouse reference genomes  
    'GRCm39': 'Male',    # C57BL/6J strain, confirmed male
    'GRCm38': 'Male',    # Same strain
    
    # Other model organisms
    'GRCz11': 'Male',    # Zebrafish Tübingen strain
    'CanFam3.1': 'Female',  # Boxer breed female
    'ARS-UCD1.2': 'Male',   # Angus bull
    'Mmul_10': 'Male'        # Rhesus macaque
}

def fetch_biosample_metadata(biosample_id):
    """
    Fetch detailed BioSample metadata
    """
    if not biosample_id or biosample_id == 'N/A':
        return {}
    
    try:
        fetch_handle = Entrez.efetch(
            db="biosample",
            id=biosample_id,
            retmode="xml"
        )
        
        xml_data = fetch_handle.read()
        fetch_handle.close()
        
        # Parse XML
        root = ET.fromstring(xml_data)
        metadata = {}
        
        # Extract basic information
        biosample = root.find('.//BioSample')
        if biosample is not None:
            metadata['biosample_id'] = biosample.get('accession', '')
            
            # Extract attributes
            attributes = biosample.findall('.//Attribute')
            for attr in attributes:
                attr_name = attr.get('attribute_name', '').lower()
                attr_value = attr.text if attr.text else ''
                
                # Map common sex-related attributes
                if attr_name in ['sex', 'gender', 'biological_sex']:
                    metadata['sex'] = attr_value
                elif attr_name in ['tissue', 'tissue_type', 'sample_type']:
                    metadata['tissue'] = attr_value
                elif attr_name in ['age', 'age_at_sampling']:
                    metadata['age'] = attr_value
                elif attr_name in ['strain', 'breed', 'cultivar']:
                    metadata['strain'] = attr_value
                elif attr_name in ['collection_site', 'geographic_location']:
                    metadata['collection_site'] = attr_value
                elif attr_name in ['dev_stage', 'developmental_stage']:
                    metadata['dev_stage'] = attr_value
        
        return metadata
        
    except Exception as e:
        print(f"Error fetching BioSample {biosample_id}: {e}")
        return {}

def parse_sex_from_text(text_fields):
    """
    Parse sex information from various text fields using regex
    """
    if not text_fields:
        return 'Unknown'
    
    # Combine all text fields
    combined_text = ' '.join(str(field) for field in text_fields if field).lower()
    
    # Sex patterns
    male_patterns = [
        r'\bmale\b', r'\bm\b(?!ouse)', r'\bxy\b', r'\btestis\b', r'\btestes\b',
        r'\bsperm\b', r'\bprostata\b', r'\bprostate\b'
    ]
    
    female_patterns = [
        r'\bfemale\b', r'\bf\b(?!ly)', r'\bxx\b', r'\bovary\b', r'\bovaries\b',
        r'\buterus\b', r'\bpregnant\b', r'\bpregnancy\b'
    ]
    
    # Check for male indicators
    for pattern in male_patterns:
        if re.search(pattern, combined_text):
            return 'Male'
    
    # Check for female indicators
    for pattern in female_patterns:
        if re.search(pattern, combined_text):
            return 'Female'
    
    return 'Unknown'

def infer_sex_from_assembly_name(assembly_name):
    """
    Use known mappings to infer sex from assembly name
    """
    if assembly_name in KNOWN_SEX_MAPPINGS:
        return KNOWN_SEX_MAPPINGS[assembly_name]
    
    # Check for partial matches
    for known_name, sex in KNOWN_SEX_MAPPINGS.items():
        if known_name.lower() in assembly_name.lower():
            return sex
    
    return 'Unknown'

def enrich_with_biosample_data(assembly_df):
    """
    Main function to enrich assembly data with BioSample metadata
    """
    print("Starting BioSample metadata enrichment...")
    
    enriched_data = []
    
    for idx, row in assembly_df.iterrows():
        print(f"\nProcessing {row['assembly_name']} ({idx+1}/{len(assembly_df)})...")
        
        # Start with assembly data
        enriched_row = row.to_dict()
        
        # Initialize sex determination
        determined_sex = 'Unknown'
        sex_source = 'Unknown'
        
        # Method 1: Check known mappings first
        inferred_sex = infer_sex_from_assembly_name(row['assembly_name'])
        if inferred_sex != 'Unknown':
            determined_sex = inferred_sex
            sex_source = 'Literature/Database'
        
        # Method 2: Fetch BioSample if available
        biosample_data = {}
        if row.get('biosample_accession') and row['biosample_accession'] != '':
            biosample_data = fetch_biosample_metadata(row['biosample_accession'])
            
            # Check if sex is in BioSample
            if 'sex' in biosample_data and biosample_data['sex']:
                determined_sex = biosample_data['sex'].title()
                sex_source = 'BioSample'
        
        # Method 3: Parse from text fields if still unknown
        if determined_sex == 'Unknown':
            text_fields = [
                row.get('assembly_name', ''),
                row.get('submitter', ''),
                biosample_data.get('strain', ''),
                biosample_data.get('tissue', '')
            ]
            parsed_sex = parse_sex_from_text(text_fields)
            if parsed_sex != 'Unknown':
                determined_sex = parsed_sex
                sex_source = 'Text Parsing'
        
        # Add enriched fields
        enriched_row['determined_sex'] = determined_sex
        enriched_row['sex_source'] = sex_source
        enriched_row['biosample_tissue'] = biosample_data.get('tissue', '')
        enriched_row['biosample_age'] = biosample_data.get('age', '')
        enriched_row['biosample_strain'] = biosample_data.get('strain', '')
        enriched_row['biosample_collection_site'] = biosample_data.get('collection_site', '')
        enriched_row['biosample_dev_stage'] = biosample_data.get('dev_stage', '')
        
        enriched_data.append(enriched_row)
        
        # Rate limiting
        time.sleep(0.5)
    
    # Create enriched DataFrame
    enriched_df = pd.DataFrame(enriched_data)
    
    # Save enriched data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enriched_filename = f"mammalian_reference_genomes_with_biosample_{timestamp}.csv"
    enriched_df.to_csv(enriched_filename, index=False)
    
    print(f"\nEnriched data saved to: {enriched_filename}")
    
    # Summary statistics
    print("\nSex Distribution Summary:")
    sex_counts = enriched_df['determined_sex'].value_counts()
    print(sex_counts)
    
    print("\nSex Source Summary:")
    source_counts = enriched_df['sex_source'].value_counts()
    print(source_counts)
    
    return enriched_df

def main():
    """
    Main function - loads assembly data and enriches it
    """
    # Look for the most recent assembly metadata file
    import glob
    import os
    
    assembly_files = glob.glob("assembly_metadata_raw_*.csv")
    if not assembly_files:
        print("No assembly metadata files found. Please run fetch_assembly_metadata.py first.")
        return None
    
    # Use the most recent file
    latest_file = max(assembly_files, key=os.path.getctime)
    print(f"Loading assembly data from: {latest_file}")
    
    assembly_df = pd.read_csv(latest_file)
    
    # Enrich with BioSample data
    enriched_df = enrich_with_biosample_data(assembly_df)
    
    return enriched_df

if __name__ == "__main__":
    enriched_data = main()

