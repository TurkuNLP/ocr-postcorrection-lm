import json
from tqdm import tqdm
import argparse
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import re

def read_metadata(file_path):
    """Read metadata from a JSONL file containing dictionaries with ecco_id and start_year."""
    metadata = {}
    with open(file_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Reading metadata"):
            entry = json.loads(line)
            ecco_id = entry.get("ecco_id")
            start_year = entry.get("start_year")
            metadata[ecco_id] = start_year
    return metadata

def read_books(file_path):
    books = []
    with open(file_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading books"):
            books.append(json.loads(line))
    return books

def terms_in_book(term, book):

    pattern = r'\b' + re.escape(term) + r'\b'
    
    # Find all matches (case-insensitive)
    before = len(re.findall(pattern, book["ecco input"], re.IGNORECASE))
    after = len(re.findall(pattern, book["ecco corrected"], re.IGNORECASE))

    return before, after

def plot_years_distribution(years, output_file, title, term_before=None, term_after=None):
    """Plot a histogram of the years distribution."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Create histogram on primary axis
    bins=max(years)-min(years)
    ax1.hist(years, bins=bins, edgecolor='black')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Number of Books', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add line plot for term_before if provided
    if term_before is not None:
        ax2 = ax1.twinx()
        # Count occurrences per year
        year_counts_before = {}
        for year in term_before:
            year_counts_before[year] = year_counts_before.get(year, 0) + 1
        # Sort years and get counts
        sorted_years = sorted(year_counts_before.keys())
        counts = [year_counts_before[year] for year in sorted_years]
        year_counts_after = {}
        for year in term_after:
            year_counts_after[year] = year_counts_after.get(year, 0) + 1
        # Sort years and get counts
        sorted_years_after = sorted(year_counts_after.keys())
        counts_after = [year_counts_after[year] for year in sorted_years_after]
        
        # Plot line
        ax2.plot(sorted_years, counts, 'r-', linewidth=2, label='Term occurrences before')
        ax2.set_ylabel('Term Count', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Plot line for term_after if provided
        if term_after is not None:
            year_counts_after = {}
            for year in term_after:
                year_counts_after[year] = year_counts_after.get(year, 0) + 1
            # Sort years and get counts
            sorted_years_after = sorted(year_counts_after.keys())
            counts_after = [year_counts_after[year] for year in sorted_years_after]
            ax2.plot(sorted_years_after, counts_after, 'g-', linewidth=2, label='Term occurrences after')
    
    # Add title and grid
    plt.title(title, fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # add legend
    if term_before is not None:
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main(args):
    # Read metadata
    metadata = read_metadata(args.metadata_file)
    
    # Print some statistics about the years
    years = list(metadata.values())
    print(f"Earliest year: {min(years)}")
    print(f"Latest year: {max(years)}")
    print(f"Average year: {sum(years)/len(years):.1f}")

    # plot this
    ecco_years = [metadata[key] for key in metadata.keys() if metadata[key] >= 1680 and metadata[key] <= 1805] # remove outliers
    plot_years_distribution(ecco_years, "ecco_books_by_years.png", "ECCO Publication Years")

    # read books
    tcp_books = read_books(args.ecco_books)
    tcp_years = [metadata[book["book_id"]] for book in tcp_books]
    term_before, term_after = [], []
    term = "publick"
    for book in tqdm(tcp_books, desc=f"Looking for a term {term}"):
        year = metadata[book["book_id"]]
        before, after = terms_in_book(term, book)
        term_before += [year]*before
        term_after += [year]*after
    plot_years_distribution(tcp_years, "tcp_books_by_years.png", "TCP Publication Years", term_before, term_after)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-file", type=str, required=True, help="Path to the JSONL file containing ecco metadata")
    parser.add_argument("--ecco-books", type=str, required=True, help="Path to the JSONL file containing ecco books")
    args = parser.parse_args()
    
    main(args) 