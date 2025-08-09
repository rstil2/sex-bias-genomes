#!/usr/bin/env python3
"""
Script 3: Quantify Sex Representation and Create Visualizations
Analyzes sex distribution in reference genomes and creates publication-ready figures.

Author: Project 33 - Bias in Reference Genomes
Date: June 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_enriched_data():
    """
    Load the most recent enriched dataset
    """
    enriched_files = glob.glob("mammalian_reference_genomes_with_biosample_*.csv")
    if not enriched_files:
        print("No enriched data files found. Please run enrich_biosample_metadata.py first.")
        return None
    
    # Use the most recent file
    latest_file = max(enriched_files, key=os.path.getctime)
    print(f"Loading enriched data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df

def clean_and_standardize_data(df):
    """
    Clean and standardize the dataset
    """
    print("Cleaning and standardizing data...")
    
    # Standardize sex categories
    sex_mapping = {
        'male': 'Male',
        'Male': 'Male',
        'MALE': 'Male',
        'M': 'Male',
        'female': 'Female', 
        'Female': 'Female',
        'FEMALE': 'Female',
        'F': 'Female',
        'unknown': 'Unknown',
        'Unknown': 'Unknown',
        'not provided': 'Unknown',
        'not collected': 'Unknown',
        '': 'Unknown',
        np.nan: 'Unknown'
    }
    
    df['sex_standardized'] = df['determined_sex'].map(sex_mapping).fillna('Unknown')
    
    # Create taxonomic groupings
    taxonomic_groups = {
        'Homo sapiens': 'Primates',
        'Pan troglodytes': 'Primates', 
        'Gorilla gorilla': 'Primates',
        'Pongo abelii': 'Primates',
        'Macaca mulatta': 'Primates',
        'Macaca fascicularis': 'Primates',
        'Mus musculus': 'Rodents',
        'Rattus norvegicus': 'Rodents',
        'Canis lupus': 'Carnivores',
        'Felis catus': 'Carnivores',
        'Bos taurus': 'Ungulates',
        'Sus scrofa': 'Ungulates',
        'Ovis aries': 'Ungulates',
        'Equus caballus': 'Ungulates'
    }
    
    df['taxonomic_group'] = df['organism_name'].map(taxonomic_groups).fillna('Other')
    
    # Add time periods based on submission dates
    df['submission_date'] = pd.to_datetime(df['submission_date'], errors='coerce')
    df['submission_year'] = df['submission_date'].dt.year
    
    # Create time periods
    def assign_period(year):
        if pd.isna(year):
            return 'Unknown'
        elif year < 2010:
            return 'Early (pre-2010)'
        elif year < 2015:
            return 'Mid (2010-2014)'
        elif year < 2020:
            return 'Recent (2015-2019)'
        else:
            return 'Latest (2020+)'
    
    df['time_period'] = df['submission_year'].apply(assign_period)
    
    return df

def calculate_summary_statistics(df):
    """
    Calculate comprehensive summary statistics
    """
    print("\nCalculating summary statistics...")
    
    # Overall sex distribution
    overall_sex = df['sex_standardized'].value_counts()
    overall_pct = df['sex_standardized'].value_counts(normalize=True) * 100
    
    print("\nOverall Sex Distribution:")
    for sex, count in overall_sex.items():
        pct = overall_pct[sex]
        print(f"  {sex}: {count} ({pct:.1f}%)")
    
    # Sex distribution by taxonomic group
    print("\nSex Distribution by Taxonomic Group:")
    tax_sex = pd.crosstab(df['taxonomic_group'], df['sex_standardized'], margins=True)
    print(tax_sex)
    
    # Sex distribution by time period
    print("\nSex Distribution by Time Period:")
    time_sex = pd.crosstab(df['time_period'], df['sex_standardized'], margins=True)
    print(time_sex)
    
    # Assembly level analysis
    print("\nSex Distribution by Assembly Level:")
    assembly_sex = pd.crosstab(df['assembly_level'], df['sex_standardized'], margins=True)
    print(assembly_sex)
    
    # Statistical tests
    print("\nStatistical Tests:")
    
    # Chi-square test for sex distribution vs taxonomic groups
    contingency_tax = pd.crosstab(df['taxonomic_group'], df['sex_standardized'])
    chi2_tax, p_tax = stats.chi2_contingency(contingency_tax)[:2]
    print(f"Sex vs Taxonomic Group - Chi-square: {chi2_tax:.3f}, p-value: {p_tax:.3e}")
    
    # Chi-square test for sex distribution vs time period
    contingency_time = pd.crosstab(df['time_period'], df['sex_standardized'])
    chi2_time, p_time = stats.chi2_contingency(contingency_time)[:2]
    print(f"Sex vs Time Period - Chi-square: {chi2_time:.3f}, p-value: {p_time:.3e}")
    
    return {
        'overall_sex': overall_sex,
        'taxonomic_sex': tax_sex,
        'time_sex': time_sex,
        'assembly_sex': assembly_sex,
        'chi2_tax': (chi2_tax, p_tax),
        'chi2_time': (chi2_time, p_time)
    }

def create_publication_figures(df, stats_dict):
    """
    Create publication-ready figures
    """
    print("\nCreating publication figures...")
    
    # Figure 1: Overall sex distribution pie chart
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Overall distribution
    sex_counts = df['sex_standardized'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    ax1.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
           colors=colors[:len(sex_counts)], startangle=90)
    ax1.set_title('Overall Sex Distribution\nof Reference Genomes', fontsize=14, fontweight='bold')
    
    # Subplot 2: Sex by taxonomic group
    tax_counts = pd.crosstab(df['taxonomic_group'], df['sex_standardized'])
    tax_counts.plot(kind='bar', stacked=True, ax=ax2, color=colors)
    ax2.set_title('Sex Distribution by Taxonomic Group', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Taxonomic Group')
    ax2.set_ylabel('Number of Reference Genomes')
    ax2.legend(title='Sex')
    ax2.tick_params(axis='x', rotation=45)
    # Fix overlapping labels
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Subplot 3: Sex by time period
    time_counts = pd.crosstab(df['time_period'], df['sex_standardized'])
    time_counts.plot(kind='bar', ax=ax3, color=colors)
    ax3.set_title('Sex Distribution by Time Period', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time Period')
    ax3.set_ylabel('Number of Reference Genomes')
    ax3.legend(title='Sex')
    ax3.tick_params(axis='x', rotation=45)
    
    # Subplot 4: Heatmap of completeness
    # Create completeness matrix
    organisms = df['organism_name'].unique()
    sex_completeness = []
    for org in organisms:
        org_data = df[df['organism_name'] == org]
        male_count = len(org_data[org_data['sex_standardized'] == 'Male'])
        female_count = len(org_data[org_data['sex_standardized'] == 'Female'])
        unknown_count = len(org_data[org_data['sex_standardized'] == 'Unknown'])
        total = len(org_data)
        
        # Create shorter organism names for better display
        short_name = org.split('(')[0].strip()  # Remove parenthetical parts
        if 'sapiens' in org:
            short_name = 'H. sapiens'
        elif 'musculus' in org:
            short_name = 'M. musculus'
        elif 'norvegicus' in org:
            short_name = 'R. norvegicus'
        elif 'mulatta' in org:
            short_name = 'M. mulatta'
        elif 'fascicularis' in org:
            short_name = 'M. fascicularis'
        elif 'troglodytes' in org:
            short_name = 'P. troglodytes'
        elif 'gorilla' in org:
            short_name = 'G. gorilla'
        elif 'abelii' in org:
            short_name = 'P. abelii'
        elif 'lupus' in org:
            short_name = 'C. lupus'
        elif 'catus' in org:
            short_name = 'F. catus'
        elif 'taurus' in org:
            short_name = 'B. taurus'
        elif 'scrofa' in org:
            short_name = 'S. scrofa'
        elif 'aries' in org:
            short_name = 'O. aries'
        elif 'caballus' in org:
            short_name = 'E. caballus'
        
        sex_completeness.append({
            'Organism': short_name,
            'Male': male_count/total if total > 0 else 0,
            'Female': female_count/total if total > 0 else 0,
            'Unknown': unknown_count/total if total > 0 else 0
        })
    
    completeness_df = pd.DataFrame(sex_completeness)
    completeness_matrix = completeness_df.set_index('Organism')[['Male', 'Female', 'Unknown']]
    
    sns.heatmap(completeness_matrix, annot=True, cmap='RdYlBu_r', ax=ax4, 
                cbar_kws={'label': 'Proportion'}, fmt='.2f')
    ax4.set_title('Sex Representation Completeness\nby Organism', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Sex Category')
    ax4.set_ylabel('')
    
    # Adjust y-axis labels to prevent overlapping
    ax4.tick_params(axis='y', labelsize=10)
    plt.setp(ax4.get_yticklabels(), rotation=0, ha='right')
    
    plt.tight_layout()
    plt.savefig('Figure1_Sex_Distribution_Analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure1_Sex_Distribution_Analysis.pdf', bbox_inches='tight')
    plt.show()
    
    # Figure 2: Interactive Plotly visualization
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sex Distribution Over Time', 'Sequencing Technology by Sex',
                       'Assembly Level by Sex', 'Sex Source Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Time series plot
    yearly_data = df.groupby(['submission_year', 'sex_standardized']).size().reset_index(name='count')
    for sex in yearly_data['sex_standardized'].unique():
        sex_data = yearly_data[yearly_data['sex_standardized'] == sex]
        fig2.add_trace(
            go.Scatter(x=sex_data['submission_year'], y=sex_data['count'], 
                      mode='lines+markers', name=f'{sex}'),
            row=1, col=1
        )
    
    # Sequencing technology
    tech_sex = pd.crosstab(df['sequencing_tech'], df['sex_standardized'])
    for sex in ['Male', 'Female', 'Unknown']:
        if sex in tech_sex.columns:
            fig2.add_trace(
                go.Bar(x=tech_sex.index, y=tech_sex[sex], name=f'{sex}'),
                row=1, col=2
            )
    
    # Assembly level
    level_sex = pd.crosstab(df['assembly_level'], df['sex_standardized'])
    for sex in ['Male', 'Female', 'Unknown']:
        if sex in level_sex.columns:
            fig2.add_trace(
                go.Bar(x=level_sex.index, y=level_sex[sex], name=f'{sex}', showlegend=False),
                row=2, col=1
            )
    
    # Sex source pie chart
    source_counts = df['sex_source'].value_counts()
    fig2.add_trace(
        go.Pie(labels=source_counts.index, values=source_counts.values),
        row=2, col=2
    )
    
    fig2.update_layout(height=800, title_text="Comprehensive Sex Bias Analysis")
    fig2.write_html("Figure2_Interactive_Analysis.html")
    
    print("Figures saved:")
    print("  - Figure1_Sex_Distribution_Analysis.png")
    print("  - Figure1_Sex_Distribution_Analysis.pdf")
    print("  - Figure2_Interactive_Analysis.html")

def generate_summary_report(df, stats_dict):
    """
    Generate a comprehensive summary report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Sex Bias in Reference Genomes - Analysis Report
Generated: {timestamp}

## Dataset Summary
- Total reference genomes analyzed: {len(df)}
- Unique organisms: {df['organism_name'].nunique()}
- Taxonomic groups: {df['taxonomic_group'].nunique()}
- Time range: {df['submission_year'].min():.0f}-{df['submission_year'].max():.0f}

## Key Findings

### Overall Sex Distribution
"""
    
    sex_counts = df['sex_standardized'].value_counts()
    sex_pct = df['sex_standardized'].value_counts(normalize=True) * 100
    
    for sex in sex_counts.index:
        report += f"- {sex}: {sex_counts[sex]} genomes ({sex_pct[sex]:.1f}%)\n"
    
    # Calculate bias metrics
    male_count = sex_counts.get('Male', 0)
    female_count = sex_counts.get('Female', 0)
    total_sexed = male_count + female_count
    
    if total_sexed > 0:
        male_bias = (male_count - female_count) / total_sexed * 100
        report += f"\n### Bias Metrics\n"
        report += f"- Male bias: {male_bias:.1f}% (positive = male-biased)\n"
        report += f"- Sex-determined genomes: {total_sexed}/{len(df)} ({total_sexed/len(df)*100:.1f}%)\n"
    
    report += f"""

### Statistical Significance
- Sex vs Taxonomic Group: χ² = {stats_dict['chi2_tax'][0]:.3f}, p = {stats_dict['chi2_tax'][1]:.3e}
- Sex vs Time Period: χ² = {stats_dict['chi2_time'][0]:.3f}, p = {stats_dict['chi2_time'][1]:.3e}

### Recommendations
1. **Immediate Actions:**
   - Prioritize female reference genomes for under-represented species
   - Implement sex-aware quality metrics in genome databases
   - Require sex annotation for all new reference submissions

2. **Long-term Strategies:**
   - Develop pan-sex reference frameworks
   - Create sex-stratified benchmarking datasets
   - Integrate sex bias correction in AI/ML pipelines

3. **Data Quality:**
   - {sex_counts.get('Unknown', 0)} genomes lack sex annotation
   - Standardize sex metadata across databases
   - Cross-reference with publication and lab records
"""
    
    # Save report
    with open('Sex_Bias_Analysis_Report.md', 'w') as f:
        f.write(report)
    
    print("\nAnalysis report saved: Sex_Bias_Analysis_Report.md")
    return report

def main():
    """
    Main analysis function
    """
    print("Starting sex representation analysis...")
    
    # Load data
    df = load_enriched_data()
    if df is None:
        return
    
    # Clean and standardize
    df_clean = clean_and_standardize_data(df)
    
    # Calculate statistics
    stats_dict = calculate_summary_statistics(df_clean)
    
    # Create figures
    create_publication_figures(df_clean, stats_dict)
    
    # Generate report
    report = generate_summary_report(df_clean, stats_dict)
    
    print("\nAnalysis complete! Files generated:")
    print("  - Sex_Bias_Analysis_Report.md")
    print("  - Figure1_Sex_Distribution_Analysis.png/.pdf")
    print("  - Figure2_Interactive_Analysis.html")
    
    return df_clean, stats_dict

if __name__ == "__main__":
    cleaned_data, statistics = main()

