
CODE REPOSITORY

This directory contains all analysis code for the "Sex Bias in Reference Genomes and AI Diagnostics" study.

FILES:

1. fetch_assembly_metadata.py
   - Fetches reference genome metadata from NCBI Assembly database
   - Requires: Biopython, pandas
   - Usage: python fetch_assembly_metadata.py

2. enrich_biosample_metadata.py
   - Enriches assembly data with BioSample metadata and sex determination
   - Requires: Output from fetch_assembly_metadata.py
   - Usage: python enrich_biosample_metadata.py

3. analyze_sex_representation.py
   - Analyzes sex distribution patterns and creates visualizations
   - Requires: Output from enrich_biosample_metadata.py
   - Usage: python analyze_sex_representation.py

4. ai_benchmarking.py
   - Benchmarks AI tools for sex-dependent bias
   - Simulates genomic data and evaluates model performance
   - Usage: python ai_benchmarking.py

5. prepare_submission.py
   - Organizes all files for manuscript submission
   - Usage: python prepare_submission.py

6. requirements.txt
   - Python package dependencies
   - Installation: pip install -r requirements.txt

SYSTEM REQUIREMENTS:
- Python 3.8+
- Internet connection for NCBI API access
- ~2GB disk space for data and figures

EXECUTION ORDER:
1. Install dependencies: pip install -r requirements.txt
2. Run data collection: python fetch_assembly_metadata.py
3. Enrich with metadata: python enrich_biosample_metadata.py
4. Analyze patterns: python analyze_sex_representation.py
5. Benchmark AI tools: python ai_benchmarking.py
6. Prepare submission: python prepare_submission.py

NOTES:
- NCBI API calls are rate-limited to be respectful to servers
- All scripts include error handling and progress reporting
- Output files are timestamped to avoid conflicts
