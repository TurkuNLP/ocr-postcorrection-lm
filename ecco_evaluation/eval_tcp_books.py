import json
import argparse
import re
import sys
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation')) # import from sibling directory
from eval_metrics import calculate_metrics

metric = None
predicted_field = None

def create_plot(results, output_file_name, outlier_value=0.4):
    """Plot old and new metrics for all books, with color indicating improvement."""

    assert metric is not None, "Metric is not set"

    # Set font sizes globally
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    results = list(results.values())

    if outlier_value == 0.0:
        outlier_value = 1000 # set to high value to keep all books

    outliers = [(book["book_id"], book[f"original {metric}"], book[f"corrected {metric}"]) for book in results if book[f"original {metric}"] > outlier_value]
    print(f"Removing outliers (original value higher than {outlier_value}): {len(outliers)} Books (id, orig {metric}, corrected {metric}): {outliers}")

    original_metric_values = [book[f"original {metric}"] for book in results if book[f"original {metric}"] < outlier_value]
    corrected_metric_values = [book[f"corrected {metric}"] for book in results if book[f"original {metric}"] < outlier_value]
    improvements = [book["improvement"]*100 for book in results if book[f"original {metric}"] < outlier_value]
    assert len(original_metric_values) == len(corrected_metric_values) == len(improvements)

    # Plot the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(corrected_metric_values, original_metric_values, c=improvements, cmap='RdYlGn', edgecolor='k', s=50, vmin=-100, vmax=100)
    ax.plot([0, max(original_metric_values)], [0, max(original_metric_values)], 'k-', label='y=x (no change)')

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Improvement (%)')

    # Labels and title
    ax.set_xlabel(f'Corrected {metric.upper()}')
    ax.set_ylabel(f'Original {metric.upper()}')
    ax.legend(bbox_to_anchor=(1, 0), loc='lower right')
    ax.set_xlim(0, max(original_metric_values) + 0.05)
    ax.set_ylim(0, max(original_metric_values) + 0.05)

    # Show plot
    plt.tight_layout()
    plt.savefig(output_file_name)
    plt.close()


def run_book_level_eval(book):

    assert metric is not None, "Metric is not set"
    assert predicted_field is not None, "Predicted field is not set"

    # we need ecco input, ecco corrected postprocessed, and tcp reference
    original = book["ecco input"]
    predicted = book[predicted_field]
    reference = book["tcp reference"]

    results = calculate_metrics(predictions=[predicted], references=[reference], originals=[original], metric=metric)

    micro_metric = results["micro"][metric]
    improvement = results["improvement"][metric]["mean"] # we calculate this separately for each book, so mean/median/w.average are the same
    original_metric = calculate_metrics(predictions=[original], references=[reference], metric=metric)["micro"][metric]

    return (book["book_id"], micro_metric, original_metric, improvement)


def run_eval(books, args):
    # evaluate each book separately (parallel), creates a dictionary where key: book id, value is a dictionary of book id, corrected metric, original metric, and improvement
    # save to a file
    print(f"Running {args.metric.upper()} evaluation for all books separately")
    results = {}
    pool = Pool(20)
    with open(args.output, "wt", encoding="utf-8") as f:
        print("\t".join(["book_id", f"corrected {args.metric}", f"original {args.metric}", "improvement"]), file=f)
        for r in tqdm(pool.imap_unordered(run_book_level_eval, books), total=len(books)):
            book_id, corrected_metric, original_metric, improvement = r
            results[book_id] = {"book_id": book_id, f"corrected {args.metric}": corrected_metric, f"original {args.metric}": original_metric, "improvement": improvement}
            print("\t".join(str(x) for x in[book_id, corrected_metric, original_metric, improvement]), file=f, flush=True)

    return results

def read_eval_results(file_name):
    print(f"Reading evaluation results from {file_name}")
    results = {}
    with open(file_name, "rt", encoding="utf-8") as f:
        for line in tqdm(f.readlines()[1:], desc="Reading evaluation results"):
            book_id, corrected_metric, original_metric, improvement = line.strip().split("\t")
            results[book_id] = {"book_id": book_id, f"corrected {args.metric}": float(corrected_metric), f"original {args.metric}": float(original_metric), "improvement": float(improvement)}
    return results


def main(args):

    # hacky way to pass metric to parallel run_book_level_eval
    global metric
    global predicted_field

    metric = args.metric
    predicted_field = args.predicted_field

    # read data
    print(f"Reading data from {args.data}")
    books = []
    with open(args.data, "rt", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Loading books"):
            books.append(json.loads(line))

    print(f"Running evaluation for {len(books)} books using field {args.predicted_field}.")

   # if file exists, read results
    if os.path.exists(args.output):
        results = read_eval_results(args.output)
    else:
        results = run_eval(books, args)

    
            
    # create a plot from the book-level results
    create_plot(results, args.plot_fname, outlier_value=args.plot_outlier_value)

    # print out the number of books that have improved and the number of books that have not improved
    improved = 0
    for key in results.keys():
        book_results = results[key]
        if book_results[f"corrected {args.metric}"] < book_results[f"original {args.metric}"]:
            improved += 1
    print("Positive improvement:", improved, "Total:", len(results))
    print("Negative improvement:", len(results) - improved, "Total:", len(results))

    # run the full evaluation metrics (mean, median, weighted average) across all books
    print("Running full evaluation.")

    # metric weights
    if args.metric == "cer":
        total_len = sum([len(b[predicted_field]) for b in books])
        weights = [len(b[predicted_field])/total_len for b in books]
    else:  # wer
        total_len = sum([len(b[predicted_field].split()) for b in books])
        weights = [len(b[predicted_field].split())/total_len for b in books]
    
    original_metric_results = [results[b["book_id"]][f"original {args.metric}"] for b in books]
    metric_results = [results[b["book_id"]][f"corrected {args.metric}"] for b in books]
    improvements = [results[b["book_id"]]["improvement"] for b in books]

    print(f"Original mean {args.metric.upper()}: {np.average(original_metric_results)}, Corrected mean {args.metric.upper()}: {np.average(metric_results)}")
    print(f"Original median {args.metric.upper()}: {np.median(original_metric_results)}, Corrected median {args.metric.upper()}: {np.median(metric_results)}")
    print(f"Original weighted average {args.metric.upper()}: {np.average(original_metric_results, weights=weights)}, Corrected weighted average {args.metric.upper()}: {np.average(metric_results, weights=weights)}")
    print(f"Weighted average {args.metric.upper()} improvements: {np.average(improvements, weights=weights)}")
    
    

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="final_ecco_tcp_joined_books.jsonl", help="Read combined TCP ECCO data.")
    parser.add_argument("--predicted-field", default="ecco corrected postprocessed", help="Field to use for evaluation.")
    parser.add_argument("--output", default="results.tsv", help="Save book-level results.")
    parser.add_argument("--plot-fname", default="plot.png", help="Plot file name.")
    parser.add_argument("--metric", default="cer", choices=["cer", "wer"], help="Metric to use for evaluation (cer or wer)")
    parser.add_argument("--plot-outlier-value", type=float, default=0.4, help="Value for discarding outlier when plotting, all books where original metric is higher than this value will be discarded. Use 0.0 for all.")
    args = parser.parse_args()

    main(args)

    # python eval_tcp_books.py --data final_ecco_tcp_joined_books.jsonl --output cer_results.jsonl --metric cer