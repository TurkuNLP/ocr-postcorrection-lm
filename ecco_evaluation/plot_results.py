import json
import matplotlib.pyplot as plt
import numpy as np


def read_data():
    with open("eval_results.jsonl", "rt", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    return results

def stats(results):
    improved = 0
    for book in results:
        orig = book["ecco input cer"]
        corrected = book["ecco corrected cer"]

        #orig = book["Original CER (ecco)"]
        #corrected = book["CER after correction"]
        #print(corrected < orig, "Book:", book["book_id"], "Orig CER:",orig, "Corrected CER:", corrected)

        if corrected < orig:
            improved += 1
        else:
            print("Book:", book["book_id"], "Orig CER:",orig, "Corrected CER:", corrected)

    print("Positive:", improved, "Total:", len(results))
    print("Negative:", len(results) - improved, "Total:", len(results))




def create_plot(results):

    outliers = [(book["book_id"], round(book["ecco input cer"],2), round(book["ecco corrected cer"],2)) for book in results if book["ecco input cer"] > 0.4]
    print(f"Outliers: {len(outliers)} Books: {outliers}")

    original_cer_values = [book["ecco input cer"] for book in results if book["ecco input cer"] < 0.4]# and book["ecco corrected cer"] < 0.4] # remove outliers
    corrected_cer_values = [book["ecco corrected cer"] for book in results if book["ecco input cer"] < 0.4]# and book["ecco corrected cer"] < 0.4] # remove outliers
    improvements = [max(min((orig - corr ) / orig * 100, 100), -100) for orig, corr in zip(original_cer_values, corrected_cer_values)]  # Improvements

    # Plot the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(corrected_cer_values, original_cer_values, c=improvements, cmap='RdYlGn', edgecolor='k', s=50)
    ax.plot([0, max(original_cer_values)], [0, max(original_cer_values)], 'k-', label='y=x (no change)')

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Improvement')

    # Labels and title
    ax.set_xlabel('CER after correction')
    ax.set_ylabel('Original CER')
    ax.legend()
    ax.set_xlim(0, max(original_cer_values) + 0.05)
    ax.set_ylim(0, max(original_cer_values) + 0.05)
    plt.title('CER Improvement Visualization')

    # Show plot
    plt.tight_layout()
    plt.show()


def main():
    results = read_data()
    stats(results)
    create_plot(results)


main()