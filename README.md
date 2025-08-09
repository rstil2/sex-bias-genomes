# Sex Bias in Reference Genomes: AI Diagnostic Performance Analysis

This repository contains all data, analysis code, and documentation for the research paper:

**"Built-In Bias: How Reference Genome Sex Composition Affects AI Diagnostic Performance"**

## Overview

This study systematically analyzes sex representation across mammalian reference genomes and evaluates the impact of sex composition disparities on AI diagnostic performance through comprehensive simulation frameworks.

## Key Findings

- 43.8% of high-quality reference genomes derive from female donors
- 37.5% from male donors  
- 18.8% lack sex annotation
- Random Forest models show 23.1% lower accuracy for female samples
- Variant calling accuracy is 9.1% lower in females compared to males

## Repository Structure

```
sex-bias-genomes/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                    # Raw and processed datasets
├── scripts/                 # Analysis and data collection scripts
├── figures/                 # Generated figures and plots
├── analysis/                # Analysis results and reports
├── docs/                    # Documentation and supplementary materials
└── simulation/              # AI performance simulation code
```

## Data Files

### `data/`
- `mammalian_reference_genomes_with_biosample_*.csv` - Curated reference genome metadata with sex annotations
- `Raw_Assembly_Metadata.csv` - Raw NCBI assembly metadata
- `Enriched_Reference_Genome_Data.csv` - Enhanced dataset with BioSample information
- `Supplementary_Table_1_Reference_Genome_Summary.csv` - Summary table for publication

## Analysis Scripts

### `scripts/`
- `fetch_assembly_metadata.py` - Retrieves reference genome metadata from NCBI
- `enrich_biosample_metadata.py` - Enhances data with BioSample sex annotations
- `analyze_sex_representation.py` - Analyzes sex distribution patterns
- `ai_benchmarking.py` - Simulates AI performance across male/female samples

## Key Analysis Features

### 1. Reference Genome Sex Determination
- **Hierarchical approach** combining multiple evidence sources
- **Literature curation** for established genomes (GRCh38, CHM13)
- **BioSample database mining** using standardized attribute fields
- **Automated text mining** for sex-indicative terms

### 2. AI Performance Simulation
- **Synthetic dataset generation**: 10,000 samples (5,000 male, 5,000 female)
- **Realistic sex-dependent variation**: 
  - 15% of features showing sex-differential expression
  - X-chromosome inactivation patterns
  - Hormonal pathway differences
  - Autosomal expression differences
- **Multiple disease phenotypes**: Equal prevalence, female-biased, and male-biased conditions
- **Algorithm comparison**: Random Forest and Logistic Regression

### 3. Performance Metrics
- **Disease prediction accuracy** across three phenotypes
- **Variant calling accuracy** comparison
- **Feature importance analysis**
- **Cross-validation robustness testing**

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Fetch reference genome metadata**:
```bash
python scripts/fetch_assembly_metadata.py
```

2. **Enrich with sex annotations**:
```bash
python scripts/enrich_biosample_metadata.py
```

3. **Analyze sex representation**:
```bash
python scripts/analyze_sex_representation.py
```

4. **Run AI performance benchmarks**:
```bash
python scripts/ai_benchmarking.py
```

## Simulation Framework

The simulation framework models realistic genomic datasets with sex-dependent characteristics:

### Dataset Parameters
- **Sample size**: 10,000 synthetic genomes (balanced male/female)
- **Feature space**: 20,000 genomic features per sample
- **Sex-differential features**: 15% with fold-change range 1.2-3.0
- **Disease prevalence ratios**: 1:1, 2:1, and 3:1 across phenotypes
- **Training bias ratios**: 70:30, 50:50, 30:70 male:female
- **Validation**: 10-fold cross-validation with stratified sampling
- **Statistical testing**: Paired t-tests with Bonferroni correction

### Machine Learning Algorithms
- **Random Forest**: 100 estimators, max depth 10
- **Logistic Regression**: L2 regularization, liblinear solver
- **Performance metrics**: Accuracy, precision, recall, F1-score

## Results Summary

### Reference Genome Analysis
- **16 mammalian species** analyzed across diverse taxonomic groups
- **Sex annotation completeness**: 85% for recent submissions (2020-2024) vs 60% for earlier submissions
- **Primary information source**: BioSample database records (68.8% of annotated genomes)

### AI Performance Disparities
- **Disease Prediction**:
  - Random Forest: 84.5% (male) vs 61.4% (female) accuracy
  - Logistic Regression: 85.2% (male) vs 56.6% (female) accuracy
- **Variant Calling**: 26.5% (male) vs 17.4% (female) accuracy
- **Consistency**: Performance gaps stable across all disease phenotypes and validation folds

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{stillwell2024sex,
  title={Built-In Bias: How Reference Genome Sex Composition Affects AI Diagnostic Performance},
  author={Stillwell, R. Craig},
  journal={Scientific Reports},
  year={2024},
  note={Submitted}
}
```

## Data Availability

- **Analysis code**: Available in this repository
- **Synthetic datasets**: Generated using documented simulation parameters
- **Raw metadata**: Obtained from publicly accessible NCBI databases
- **Complete documentation**: Parameter specifications, model architectures, and validation protocols included

## Contributing

This repository supports reproducible research. All simulation parameters, model architectures, and analysis protocols are documented to ensure full reproducibility.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this research or code, please contact:
- **R. Craig Stillwell** - craig.stillwell@gmail.com
- **Institution**: University of Kentucky

## Acknowledgments

We thank the Genome Reference Consortium, Telomere-to-Telomere Consortium, and other organizations for their commitment to open genomic data. We acknowledge the importance of diverse, representative genomic resources for advancing precision medicine.
