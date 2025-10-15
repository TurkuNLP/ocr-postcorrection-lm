#!/usr/bin/env python3
# a script to read collocations from three separate files and analyze them
# 1. Read data from tsv, take only significant collocations (mi > 3), tcp serves as gold standard
# 2. Calculate precision and recall of orig/post against gold standard
# 3. Make a union/intersection graph of orig/post collocations, and for each group (intersection, only post, only orig), print the 100 most significant collocations.

import argparse
import sys
import matplotlib.pyplot as plt

def read_collocations(file_path, mi_threshold=3.0):
    """Read collocations from TSV file and filter by MI threshold."""
    collocations = {}
    identical = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word1, word2 = parts[0], parts[1]
            if word1 == word2:
                identical += 1
                continue
            mi_score = float(parts[2])
            collocate_freq = int(parts[3])
            freq1 = int(parts[4])
            freq2 = int(parts[5])
            # Normalize pair to alphabetical order and align frequencies to the new order
            pair_sorted = tuple(sorted((word1, word2)))
            if pair_sorted != (word1, word2):
                freq1, freq2 = freq2, freq1
            if mi_score >= mi_threshold:
                collocations[pair_sorted] = {
                    'mi': mi_score,
                    'collocate_freq': collocate_freq,
                    'word1_freq': freq1,
                    'word2_freq': freq2
                } 
    print(f"Removed {identical} identical collocations.")
    return collocations

def calculate_precision_recall(predicted, gold_standard):
    """Calculate precision and recall of predicted collocations against gold standard."""
    if not predicted:
        return 0.0, 0.0
    
    if not gold_standard:
        return 0.0, 0.0
    
    # Convert to sets of collocate pairs
    predicted_pairs = set(predicted.keys())
    gold_pairs = set(gold_standard.keys())
    
    # Calculate intersection
    true_positives = len(predicted_pairs.intersection(gold_pairs))
    
    # Calculate precision and recall
    precision = true_positives / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positives / len(gold_pairs) if gold_pairs else 0.0
    
    return precision, recall

def analyze_union_intersection(predicted_collocs, tcp_collocs):
    """Analyze union and intersection of original, post-processed, and TCP collocations."""
    
    predicted_pairs = set(predicted_collocs.keys())
    tcp_pairs = set(tcp_collocs.keys())
    
    # Calculate sets
    intersection = predicted_pairs.intersection(tcp_pairs)
    only_predicted = predicted_pairs - tcp_pairs
    only_tcp = tcp_pairs - predicted_pairs  # Only in TCP, not in either method
    union = predicted_pairs.union(tcp_pairs)
    
    print(f"\n=== Union/Intersection Analysis ===")
    print(f"Total unique collocations in predicted: {len(predicted_pairs)}")
    print(f"Total unique collocations in TCP (gold standard): {len(tcp_pairs)}")
    print(f"Intersection (both methods): {len(intersection)}")
    print(f"Only in predicted: {len(only_predicted)}")
    print(f"Only in TCP: {len(only_tcp)}")
    print(f"Union (total unique): {len(union)}")
    
    return intersection, only_predicted, only_tcp


def print_top_collocations(collocations, collocate_pairs, n=100):
    """Get top N collocations by MI score from a set of collocate pairs."""
    if not collocate_pairs:
        return []
    filtered = [
        (pair, collocations[pair]) for pair in collocate_pairs if pair in collocations
    ]
    sorted_collocs = sorted(filtered, key=lambda x: x[1]['mi'], reverse=True)
    for (word1, word2), data in sorted_collocs[:n]:
        print(f"{word1:<20} {word2:<20} {data['mi']:>10,.2f} {data['collocate_freq']:>10,} {data['word1_freq']:>10,} {data['word2_freq']:>10,}")



def create_venn_diagram(orig_collocs, post_collocs, tcp_collocs, output_file="collocations_venn.png"):
    """Create side-by-side Venn diagrams showing TCP vs Original and TCP vs Post-processed."""
    from matplotlib_venn import venn2
    
    # Font size parameters - change these values to adjust all text in the Venn diagram
    label_fontsize = 24
    number_fontsize = 24
    
    tcp_pairs = set(tcp_collocs.keys())
    orig_pairs = set(orig_collocs.keys())
    post_pairs = set(post_collocs.keys())
    
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # TCP vs Original
    venn2([tcp_pairs, orig_pairs], ('', ''), ax=ax1)
    
    # TCP vs Post-processed
    venn2([tcp_pairs, post_pairs], ('', ''), ax=ax2)
    
    # Set custom colors for the Venn diagrams
    for ax in [ax1, ax2]:
        patches = ax.patches
        if len(patches) >= 3:  # Venn2 creates 3 patches: left, right, intersection
            # Left circle (TCP)
            patches[0].set_facecolor('lightblue')
            patches[0].set_alpha(0.7)
            # Right circle (Original/Post-processed)
            patches[1].set_facecolor('lightcoral')
            patches[1].set_alpha(0.7)
            # Intersection
            patches[2].set_facecolor('lightgreen')
            patches[2].set_alpha(0.7)
    
    # Add custom text labels above the numbers
    # For TCP vs Original
    if len(ax1.patches) >= 3:
        # Find the text objects and position labels above them
        for text_obj in ax1.texts:
            if text_obj.get_text().isdigit():  # This is a number
                x, y = text_obj.get_position()
                # Format the number with thousands separator
                formatted = f"{int(text_obj.get_text()):,}"
                text_obj.set_text(formatted)
                text_obj.set_fontsize(number_fontsize)
                # Position label above the number
                if x < -0.3:  # Left side - TCP
                    ax1.text(x, y + 0.05, 'TCP', ha='center', va='bottom', fontsize=label_fontsize, fontweight='bold')
                elif x > 0.3:  # Right side - Original
                    ax1.text(x, y + 0.05, 'Original', ha='center', va='bottom', fontsize=label_fontsize, fontweight='bold')
                else:  # Center - Intersection
                    ax1.text(x, y + 0.05, 'TCP ∩ Original', ha='center', va='bottom', fontsize=label_fontsize, fontweight='bold')
    
    # For TCP vs Corrected
    if len(ax2.patches) >= 3:
        # Find the text objects and position labels above them
        for text_obj in ax2.texts:
            if text_obj.get_text().isdigit():  # This is a number
                x, y = text_obj.get_position()
                # Format the number with thousands separator
                formatted = f"{int(text_obj.get_text()):,}"
                text_obj.set_text(formatted)
                text_obj.set_fontsize(number_fontsize)
                # Position label above the number
                if x < -0.3:  # Left side - TCP
                    ax2.text(x, y + 0.05, 'TCP', ha='center', va='bottom', fontsize=label_fontsize, fontweight='bold')
                elif x > 0.3:  # Right side - Corrected
                    ax2.text(x, y + 0.05, 'Corrected', ha='center', va='bottom', fontsize=label_fontsize, fontweight='bold')
                else:  # Center - Intersection
                    ax2.text(x, y + 0.05, 'TCP ∩ Corrected', ha='center', va='bottom', fontsize=label_fontsize, fontweight='bold')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Side-by-side Venn diagrams saved to {output_file}")

def main(args):
    
    
    print("Reading collocation files...")
    
    # Read all three collocation files
    orig_collocs = read_collocations(args.orig, args.mi_threshold)
    post_collocs = read_collocations(args.post, args.mi_threshold)
    tcp_collocs = read_collocations(args.tcp, args.mi_threshold)
    
    print(f"Loaded {len(orig_collocs)} significant collocations from original file")
    print(f"Loaded {len(post_collocs)} significant collocations from post-processed file")
    print(f"Loaded {len(tcp_collocs)} significant collocations from TCP file (gold standard)")
    
    # Validate that we have data to work with
    if not orig_collocs or not post_collocs or not tcp_collocs:
        print(f"Error: No significant collocations found. Please check the input files and thresholds.")
        sys.exit(1)
    
    # Calculate precision and recall
    print("\n=== Precision and Recall Analysis ===")
    print("Using TCP collocations as gold standard")
    
    orig_precision, orig_recall = calculate_precision_recall(orig_collocs, tcp_collocs)
    post_precision, post_recall = calculate_precision_recall(post_collocs, tcp_collocs)
    
    print(f"Original collocations:")
    print(f"  Precision: {orig_precision:.4f} ({orig_precision*100:.1f}%)")
    print(f"  Recall: {orig_recall:.4f} ({orig_recall*100:.1f}%)")
    print(f"  F1-Score: {2 * (orig_precision * orig_recall) / (orig_precision + orig_recall):.4f}" if (orig_precision + orig_recall) > 0 else "  F1-Score: 0.0000")
    
    print(f"Post-processed collocations:")
    print(f"  Precision: {post_precision:.4f} ({post_precision*100:.1f}%)")
    print(f"  Recall: {post_recall:.4f} ({post_recall*100:.1f}%)")
    print(f"  F1-Score: {2 * (post_precision * post_recall) / (post_precision + post_recall):.4f}" if (post_precision + post_recall) > 0 else "  F1-Score: 0.0000")
    

    # Create venn visualization
    create_venn_diagram(orig_collocs, post_collocs, tcp_collocs, args.output)
    print(f"\nVenn diagram complete! Results saved to {args.output}")

    # Print examples
    print(f"\nTCP vs Original:")
    intersection1, only_orig, only_tcp1 = analyze_union_intersection(orig_collocs, tcp_collocs)
    print(f"\nTCP vs Corrected:")
    intersection2, only_post, only_tcp2 = analyze_union_intersection(post_collocs, tcp_collocs)

    # Print globally sorted top collocations per group by MI score
    print(f"\nExamples from different subsets:")

    
    print(f"\nIn all three subsets:")
    all_three = intersection1.intersection(intersection2)
    print_top_collocations(tcp_collocs, all_three, args.top_n)

    print(f"\nOnly in TCP (not in original or corrected):")
    only_tcp = only_tcp1.intersection(only_tcp2)
    print_top_collocations(tcp_collocs, only_tcp, args.top_n)

    print(f"\nOnly in original (not TCP or corrected):")
    orig = only_orig - only_post
    print_top_collocations(orig_collocs, orig, args.top_n)

    print(f"\nOnly in corrected (not TCP or original):")
    post = only_post - only_orig
    print_top_collocations(post_collocs, post, args.top_n)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze collocations from multiple files')
    parser.add_argument('--orig', default='collocations_orig.tsv', help='Original collocations file')
    parser.add_argument('--post', default='collocations_post.tsv', help='Post-processed collocations file')
    parser.add_argument('--tcp', default='collocations_tcp.tsv', help='TCP (gold standard) collocations file')
    parser.add_argument('--mi-threshold', type=float, default=3.0, help='MI threshold for significant collocations')
    parser.add_argument('--output', default='collocations_venn.png', help='Output visualization file')
    parser.add_argument('--top-n', type=int, default=100, help='Number of top collocations to display')

    
    args = parser.parse_args()
    main(args)

