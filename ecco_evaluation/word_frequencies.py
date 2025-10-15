from collections import Counter
from tqdm import tqdm
import json
import unicodedata
import re
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text, lowercase=False):

    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"-", "", text)
    text = re.sub(r"[\.\,\?\!\:\;]", "", text)
    text = re.sub(r"\s+", " ", text)

    if lowercase:
        return unicodedata.normalize("NFKC", text.casefold())
    return unicodedata.normalize("NFKC", text)



def word_frequncies(books, key):

    word_counter = Counter()

    for book in tqdm(books, desc=f"Calculating word frequencies for {key}"):
        text = book[key]
        text = normalize(text).split()
        word_counter.update(text)

    return word_counter

def analyze_changes_in_vocabularies(tcp_frequencies, original_frequencies, corrected_frequencies):
    ## Removed words
    removed_words = set(original_frequencies.keys()) - set(corrected_frequencies.keys()) # words present in orginal ecco but not in corrected ecco
    print(f"\nRemoved words: {len(removed_words)}")

    
    # % of removed words which are singletons in original ecco
    singleton_removed_words = [word for word in removed_words if original_frequencies[word] == 1]
    print(f"% of removed words which are singletons in original ecco: {len(singleton_removed_words) / len(removed_words) * 100:.2f}%")
    # % of removed words which do not appear in tcp
    removed_words_not_in_tcp = [word for word in removed_words if word not in tcp_frequencies]
    print(f"% of removed words which do not appear in tcp: {len(removed_words_not_in_tcp) / len(removed_words) * 100:.2f}%")

    ## New words
    new_words = set(corrected_frequencies.keys()) - set(original_frequencies.keys()) # words present in corrected ecco but not in original ecco
    print(f"\nNew words: {len(new_words)}")

    # % of new words which are singletons in corrected ecco
    singleton_new_words = [word for word in new_words if corrected_frequencies[word] == 1]
    print(f"% of new words which are singletons in corrected ecco: {len(singleton_new_words) / len(new_words) * 100:.2f}%")
    # % of new words which do not appear in tcp
    new_words_not_in_tcp = [word for word in new_words if word not in tcp_frequencies]
    print(f"% of new words which do not appear in tcp: {len(new_words_not_in_tcp) / len(new_words) * 100:.2f}%")

    # most frequent words in original ecco which do not appear at all in corrected ecco, including tcp count
    print("\nMost frequent (in original) removed words:")
    new_counter = Counter({word: original_frequencies[word] for word in removed_words})
    for word, count in new_counter.most_common(10):
        print(f"{word}: {count:,} (tcp count: {tcp_frequencies[word]:,})")
   

    # words which do not occur in corrected ecco but are relatively frequent in original and tcp
    print("\nMost common tcp words which are removed by correction:")
    removed_words_with_tcp_frequencies = Counter({word: tcp_frequencies[word] for word in removed_words})
    for word, count in removed_words_with_tcp_frequencies.most_common(20):
        print(f"{word}: {original_frequencies[word]:,} (tcp count: {count:,})")

    # most frequent new words
    print("\nMost frequent (in corrected) new words:")
    new_counter = Counter({word: corrected_frequencies[word] for word in new_words})
    for word, count in new_counter.most_common(10):
        print(f"{word}: {count:,} (tcp count: {tcp_frequencies[word]:,})")



def analyse_frequency_changes(tcp_frequencies, original_frequencies, corrected_frequencies):
    # the words which highest/lowest difference in raw frequency
    diffs = {}
    for word in set(original_frequencies.keys()) | set(corrected_frequencies.keys()):
        diff = corrected_frequencies[word] - original_frequencies[word]
        diffs[word] = diff

    # print top 10 words with highest difference
    print("\nTop 10 words with highest increase in raw frequency:")
    for word, diff in sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{word}: {original_frequencies[word]:,} / {corrected_frequencies[word]:,} --> {diff:,}")
    print("\nTop 10 words with highest decrease in raw frequency:")
    for word, diff in sorted(diffs.items(), key=lambda x: x[1])[:10]:
        print(f"{word}: {original_frequencies[word]:,} / {corrected_frequencies[word]:,} --> {diff:,}")

    
def print_freq(name, counter, top=15):

    print(f"Word frequencies for {name}")
    for word, freq in counter.most_common(top):
        print(f"{word}: {freq:,}")
    print()


def tfidf(books, key, top=15):
    book_texts = [book[key] for book in books]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(book_texts)
    idf_weights = vectorizer.idf_
    # print top 15 words with their idf weights
    print(f"Top {top} words with highest IDF for {key}")
    # Create sorted list of (word, idf) pairs
    d = {}
    for word, idf in zip(vectorizer.get_feature_names_out(), idf_weights):
        d[word] = idf
    return d

def select_words_to_show(tcp_frequencies, tcp_idf, number_of_words=100):
    words = []
    for word, count in tcp_frequencies.most_common(200+number_of_words):
        if word.lower() not in tcp_idf:
            continue
        if tcp_idf[word.lower()] > 5 or tcp_idf[word.lower()] < 2:
            continue
        words.append(word)

    return words


def bow(prediction, reference, lowercase=False):
    reference_tokens = normalize(reference, lowercase=lowercase).split()
    prediction_tokens = normalize(prediction, lowercase=lowercase).split()
    reference_counter = Counter(reference_tokens)
    prediction_counter = Counter(prediction_tokens)
    tp = (reference_counter & prediction_counter).total()
    fp = (prediction_counter - reference_counter).total()
    fn = (reference_counter - prediction_counter).total()

    f = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0

    return tp, fp, fn, f



def document_level_bow(books):

    d = {}

    o_scores = []
    c_scores = []

    for book in books:
        o_tp, o_fp, o_fn, o_f = bow(book["ecco input"], book["tcp reference"])
        c_tp, c_fp, c_fn, c_f = bow(book["ecco corrected"], book["tcp reference"])
        print(f"Book: {book['book_id']} orig F1: {o_f:.2f}, corrected F1: {c_f:.2f}")
        o_scores.append(o_f)
        c_scores.append(c_f)
        d[book["book_id"]] = {"orig f1":  o_f, "corrected f1": c_f}

    print("\nMacro average F1:")
    print(f"Original: {sum(o_scores) / len(o_scores):.2f}")
    print(f"Corrected: {sum(c_scores) / len(c_scores):.2f}")

    return d


def main(args):

    print(f"Reading data from {args.data}")
    books = []
    with open(args.data, "rt", encoding="utf-8") as f:
        for line in tqdm(f.readlines(), desc="Loading books"):
            books.append(json.loads(line))
    #books=books[:10]
    # calculate raw word frequencies
 
    tcp_frequencies = word_frequncies(books, "tcp reference")
    original_frequencies = word_frequncies(books, "ecco input")
    corrected_frequencies = word_frequncies(books, "ecco corrected postprocessed")

    # Vocabulary size
    print("\nVocabulary sizes:")
    print(f"TCP reference vocabulary size: {len(tcp_frequencies):,}")
    print(f"Original vocabulary size: {len(original_frequencies):,}")
    print(f"Corrected vocabulary size: {len(corrected_frequencies):,}")

    # analyze diappearing and new words (vocabulary level)
    analyze_changes_in_vocabularies(tcp_frequencies, original_frequencies, corrected_frequencies)
    
    # analyse frequency changes
    analyse_frequency_changes(tcp_frequencies, original_frequencies, corrected_frequencies)

    exit(1)

    print("\n Calculating IDF weights:")
    tcp_idf = tfidf(books, "tcp reference") # returns dictionary of word -> idf

    #original_idf = tfidf(books, "ecco input")
    #corrected_idf = tfidf(books, "ecco corrected")

    words_to_show = select_words_to_show(tcp_frequencies, tcp_idf)
    words_to_show = words_to_show + "freedom luxury antient ancient poor publick public spectacles".split()
    print(f"Selected words to show: {words_to_show}")

    for w in words_to_show:
        w = w.lower()
        print(f" IDF: {w}: {tcp_idf.get(w, 0.0):.2f}")
        print(f" TCP: {w}: {tcp_frequencies[w]:,}")
        
        print(f"ORIG: {w}: {original_frequencies[w]:,} ({'+' if original_frequencies[w] > tcp_frequencies[w] else ''}{original_frequencies[w] - tcp_frequencies[w]:,})")
        print(f"CORR: {w}: {corrected_frequencies[w]:,} ({'+' if corrected_frequencies[w] > tcp_frequencies[w] else ''}{corrected_frequencies[w] - tcp_frequencies[w]:,})")
        print()

    #print("\nWord frequencies:")
    #print_freq("TCP reference", tcp_frequencies)
    #print_freq("Original", original_frequencies)
    #print_freq("Corrected", corrected_frequencies)


    exit(1)
    # calculate bow scores
    


    #input_counter, corrected_counter, reference_counter = word_frequncies(books)
    #print_freq("ecco input", input_counter)
    #print_freq("ecco corrected", corrected_counter)
    #print_freq("tcp reference", reference_counter)

    #print(f"Vocabulary sizes:")
    #print(f"ecco input: {len(input_counter):,}")
    #print(f"ecco corrected: {len(corrected_counter):,}")
    #print(f"tcp reference: {len(reference_counter):,}")

    print("\n\n")

    f1_scores = document_level_bow(books)

    # save as tsv
    keys = ["book_id", "tcp input cer", "ecco input cer", "ecco corrected cer", "TCP-ECCO match", "tcp input chars", "ecco input chars", "tcp reference chars", "ecco corrected chars", "orig f1", "corrected f1"]
    if args.cer_data:
        with open(args.cer_data, "rt", encoding="utf-8") as finput:
            with open("book_eval_metrics.csv", "wt", encoding="utf-8") as foutput:
                print("\t".join(keys), file=foutput)
                for line in finput:
                    book = json.loads(line)
                    #if book["book_id"] not in f1_scores:
                    #    continue
                    book.update(f1_scores[book["book_id"]])

                    print("\t".join([str(book[key]) for key in keys]), file=foutput)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="final_ecco_tcp_joined_books.jsonl", help="Path to the jsonl file which stores the final joined data.")
    args = parser.parse_args()

    main(args)