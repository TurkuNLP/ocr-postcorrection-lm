# a script to calculate collocations in a corpus
# 1. Read data from jsonl
# 2. Calculate collocations using MI with frequency cutoff of 4
# MI = log({(F_{wc} * N) / (F_w * F_c * S)}) / log(2)
# store results as tsv, collocate pair and MI score

import argparse
import json
import os
import re
import string
import sys
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tqdm

def yield_books(jsonl_file, text_field):
    with open(jsonl_file, "rt", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            text = book[text_field]
            yield text


def normalize(text):

    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"-", "", text)
    text = re.sub(r'[\.\,\?\!\:\;\"]', "", text)
    text = re.sub(r"\s+", " ", text)

    return unicodedata.normalize("NFKC", text.casefold())

            

def calculate_collocations(args, selected_words):

    word_frequencies = Counter() # word -> frequency
    collocate_frequencies = Counter() # (word1, word2) -> frequency, word1 must be one of the selected words, word2 can be any co-occurance pair
    mi_scores = Counter() # (word1, word2) -> MI score

    window_size = 2
    for book_text in tqdm.tqdm(yield_books(args.data, args.text_field), desc="Calculating collocations."):
        tokens = normalize(book_text).split()
        for i in range(len(tokens)): # iterate all tokens
            word1 = tokens[i]
            word_frequencies[word1] += 1
            if args.selected_words and word1 not in selected_words:
                continue
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)): # iterate over window +/- 4
                if j == i:
                    continue
                word2 = tokens[j]
                collocate_frequencies[(word1, word2)] += 1

    # total number of tokens in the corpus
    N = sum(word_frequencies.values())

    # calculate MI for each collocate pair
    for (word1, word2), frequency in collocate_frequencies.items():
        # word frequency cutoff, do not calculate mi if either word has frequency < args.frequency_cutoff
        if word_frequencies[word1] < args.frequency_cutoff or word_frequencies[word2] < args.frequency_cutoff or frequency < args.frequency_cutoff:
            continue

        mi2 = np.log2(frequency**2 / (word_frequencies[word1] * word_frequencies[word2] / N ))

        #mi = (np.log2(frequency * N / (word_frequencies[word1] * word_frequencies[word2] * (window_size*2)))) / np.log10(2)

        if (word2, word1) in mi_scores:
            if mi_scores[(word2, word1)] != mi2:
                print(f"Warning: MI score for {(word2, word1)} is different from {(word1, word2)}: {mi_scores[(word2, word1)]} != {mi2}")
            continue

        mi_scores[(word1, word2)] = mi2

    # sort mi_scores by MI score
    mi_scores = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

    # store results as tsv, collocate pair and MI score
    significant = 0
    with open(args.output, "wt", encoding="utf-8") as f:
        for (word1, word2), mi in mi_scores:
            print(word1, word2, mi, collocate_frequencies[(word1, word2)], word_frequencies[word1], word_frequencies[word2], file=f)
            if mi >= 3.0:
                significant += 1
    print(f"Number of significant positive collocate pairs: {significant}")
    print(f"Total number of words in the corpus: {N}")


    






def word_frequncies(args):

    word_counter = Counter()

    for book_text in tqdm.tqdm(yield_books(args.data, "tcp reference"), desc="Calculating TCP word frequencies."):
        tokens = normalize(book_text).split()
        word_counter.update(tokens)

    print(f"Total number of tokens in TCP: {sum(word_counter.values())}")
    print("Saving top 50,000 words to file tcp_word_frequencies.tsv")
    with open("tcp_word_frequencies.tsv", "wt", encoding="utf-8") as f:
        for word, frequency in word_counter.most_common(50000):
            print(word, frequency, file=f)





def main(args):

    # select few high frequency words, few mid frequency words, and few low frequency words
    if not os.path.exists("tcp_word_frequencies.tsv"):
        word_frequncies(args)
    else:
        print("Word frequencies file already exists, skipping word frequency calculation.")
    # high: government (rank: 360, count: 18,941)
    # mid:  communication (rank: 3633, count: 1742)
    # low:  aristocracy (rank: 10,006 count: 452)
    selected_words = set(["government", "communication", "aristocracy"])
    if args.selected_words and not selected_words:
        print("No selected words, provide a list of sample words.")
        exit(1)

    calculate_collocations(args, selected_words)

    




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../ecco-paper-data/final_ecco_tcp_joined_books.jsonl", help="Path to the jsonl file which stores the final joined data.")
    parser.add_argument("--text_field", required=True, type=str, choices=["tcp reference", "ecco input", "ecco corrected postprocessed"], help="Field in the jsonl file which stores the text.")
    parser.add_argument("--frequency_cutoff", type=int, default=4, help="Word frequency cutoff for collocate pairs.")
    parser.add_argument("--output", type=str, default="collocations.tsv", help="Path to the output tsv file.")
    parser.add_argument("--selected_words", action="store_true", help="Use only selected words from tcp_word_frequencies.tsv.")
    args = parser.parse_args()

    main(args)