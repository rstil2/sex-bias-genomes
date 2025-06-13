#!/usr/bin/env python3
"""
Script 1: Fetch Assembly Metadata
Extracts reference genome metadata from NCBI Assembly database
for mammalian species using Entrez E-utilities.

Author: Project 33 - Bias in Reference Genomes
Date: June 2025
"""

import time
import csv
import json
from Bio import Entrez
import pandas as pd
from datetime import datetime
import requests
from xml.etree import ElementTree as ET

# Set your email for NCBI Entrez
Entrez.email = "research.project@example.edu"  # Research project email

# Target organisms (focusing on mammals initially, then expanding)
TARGET_ORGANISMS = [
    "Homo sapiens",
    "Mus musculus", 
    "Rattus norvegicus",
    "Macaca mulatta",
    "Macaca fascicularis", 
    "Pan troglodytes",
    "Gorilla gorilla",
    "Pongo abelii",
    "Canis lupus",
    "Felis catus",
    "Bos taurus",
    "Sus scrofa",
    "Ovis aries",
    "Equus caballus"
]

def search_assemblies(organism, assembly_level="chromosome"):
    """
    Search for assemblies of a given organism
    """
    search_term = f'{organism}[organism] AND ({assembly_level}[filter] OR "reference genome"[filter])'
    
    try:
        search_handle = Entrez.esearch(
            db="assembly",
            term=search_term,
            retmax=50,
            sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        return search_results['IdList']
    except Exception as e:
        print(f"Error searching assemblies for {organism}: {e}")
        return []

def fetch_assembly_summary(assembly_ids):
    """
    Fetch detailed summary information for assembly IDs
    """
    if not assembly_ids:
        return []
    
    try:
        # Convert IDs to comma-separated string
        id_string = ','.join(assembly_ids)
        
        summary_handle = Entrez.esummary(
            db="assembly",
            id=id_string,
            report="full"
        )
        summaries = Entrez.read(summary_handle, validate=False)
        summary_handle.close()
        
        return summaries['DocumentSummarySet']['DocumentSummary']
    except Exception as e:
        print(f"Error fetching assembly summaries: {e}")
        return []

def parse_assembly_metadata(summary):
    """
    Parse assembly summary into structured metadata
    """
    metadata = {
        'assembly_accession': summary.get('AssemblyAccession', ''),
        'assembly_name': summary.get('AssemblyName', ''),
        'organism_name': summary.get('Organism', ''),
        'taxid': summary.get('Taxid', ''),
        'assembly_level': summary.get('AssemblyLevel', ''),
        'submitter': summary.get('SubmitterOrganization', ''),
        'biosample_accession': summary.get('BioSampleAccn', ''),
        'bioproject_accession': summary.get('BioProjectAccn', ''),
        'submission_date': summary.get('SubmissionDate', ''),
        'release_date': summary.get('LastUpdateDate', ''),
        'sequencing_tech': summary.get('SequencingTechnology', ''),
        'assembly_type': summary.get('AssemblyType', ''),
        'genome_representation': summary.get('GenomeRepresentation', ''),
        'refseq_category': summary.get('RefSeq_category', ''),
        'ncbi_annotation': summary.get('AnnotationProvider', ''),
        'contig_n50': summary.get('ContigN50', ''),
        'scaffold_n50': summary.get('ScaffoldN50', '')
    }
    
    return metadata

def main():
    """
    Main function to collect assembly metadata
    """
    print("Starting assembly metadata collection...")
    print(f"Target organisms: {len(TARGET_ORGANISMS)}")
    
    all_metadata = []
    
    for organism in TARGET_ORGANISMS:
        print(f"\nProcessing {organism}...")
        
        # Search for assemblies
        assembly_ids = search_assemblies(organism)
        print(f"Found {len(assembly_ids)} assemblies")
        
        if assembly_ids:
            # Fetch detailed metadata
            summaries = fetch_assembly_summary(assembly_ids)
            
            for summary in summaries:
                metadata = parse_assembly_metadata(summary)
                all_metadata.append(metadata)
                print(f"  - {metadata['assembly_name']} ({metadata['assembly_accession']})")
        
        # Rate limiting to be respectful to NCBI
        time.sleep(1)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_metadata)
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"assembly_metadata_raw_{timestamp}.csv"
    df.to_csv(raw_filename, index=False)
    
    print(f"\nCollected metadata for {len(all_metadata)} assemblies")
    print(f"Raw data saved to: {raw_filename}")
    
    # Display summary
    print("\nSummary by organism:")
    summary_stats = df.groupby('organism_name').agg({
        'assembly_accession': 'count',
        'assembly_level': lambda x: x.value_counts().to_dict()
    }).round(2)
    print(summary_stats)
    
    return df

if __name__ == "__main__":
    assembly_data = main()

