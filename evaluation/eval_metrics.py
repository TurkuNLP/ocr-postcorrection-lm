from evaluate import load
import unicodedata
import argparse
import json
import numpy as np


def normalize(text: str, lowercase=False):

    if lowercase:
        return unicodedata.normalize("NFKC", text.casefold())

    return unicodedata.normalize("NFKC", text)

def calculate_metrics(*, predictions, references, metric="cer", lowercase=False):

    references = [normalize(r, lowercase) for r in references]
    predictions = [normalize(p, lowercase) for p in predictions]

    scores = {"micro": {}, "mean": {}, "median": {}}

    # micro
    if metric == "cer" or metric == "all":
        cer_metric = load("cer")
        scores["micro"]["cer"] = cer_metric.compute(predictions=predictions, references=references)
        document_scores = []
        for p, r in zip(predictions, references):
            document_scores.append(cer_metric.compute(predictions=[p], references=[r]))
        scores["mean"]["cer"] = np.mean(document_scores)
        scores["median"]["cer"] = np.median(document_scores)
    
    if metric == "wer" or metric == "all":
        wer_metric = load("wer")
        scores["micro"]["wer"] = wer_metric.compute(predictions=predictions, references=references)
        document_scores = []
        for p, r in zip(predictions, references):
            document_scores.append(wer_metric.compute(predictions=[p], references=[r]))
        scores["mean"]["wer"] = np.mean(document_scores)
        scores["median"]["wer"] = np.median(document_scores)
    
    if metric == "character" or metric == "all":
        character_metric = load("character")
        scores["micro"]["character"] = character_metric.compute(predictions=predictions, references=references)["cer_score"]
        document_scores = []
        for p, r in zip(predictions, references):
            document_scores.append(cer_metric.compute(predictions=[p], references=[r])["cer_score"])
        scores["mean"]["character"] = np.mean(document_scores)
        scores["median"]["character"] = np.median(document_scores)

    
    return scores

def evaluate(*, p_args, r_args, metric='cer', lowercase=False):
    """
    Evaluates the similarity of two lists of examples with the given metric(s) (defaults to CER). Returns a dictionary holding the score(s).

    p_args: Expects either a string holding a file name/path to the predictions, or a list where list[0] is the file name/path, and list[1] is the key to the predictions (defaults to 'output'.)
    
    r_args: Expects either a string holding a file name/path to the references, or a list where list[0] is the file name/path, and list[1] is the key to the references (defaults to 'output'.)
    
    metric: accepts the following strings: 'cer', 'wer', 'character', 'all'.
    """
    if type(p_args) != list: p_args = [p_args]
    if type(r_args) != list: r_args = [r_args]
    assert metric in ("cer", "wer", "character", "all"), "Use a valid metric. Expected either 'cer', 'wer', 'character' or 'all'."
    
    predictions = read_jsonl(p_args)
    references = read_jsonl(r_args)

    assert len(predictions) == len(references)
    metrics = calculate_metrics(predictions=predictions, references=references, metric=metric, lowercase=lowercase)

    return metrics

def read_jsonl(args):

    if len(args) == 2:
        key = args[1]
    else: key = "output"

    examples = []
    with open(args[0], "rt", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example[key])
    
    return examples

def initialize_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--predictions",
                        type=str,
                        required=True,
                        nargs="+",
                        help="REQUIRED: the name of / path to the predictions file. OPTIONAL: the key to the predictions (defaults to 'output').")
    parser.add_argument("-r",
                        "--references",
                        type=str,
                        required=True,
                        nargs="+",
                        help="REQUIRED: the name of / path to the references file. OPTIONAL: the key to the references (defaults to 'output').")
    parser.add_argument("-m",
                        "--metrics",
                        choices=["cer", "wer", "character", "all"],
                        required=False,
                        default="cer",
                        help="Choose which metric(s) to use for evaluation. If no metric is given, CER (Character Error Rate) is used.")
    parser.add_argument("-l",
                        "--lower",
                        required=False,
                        action='store_true',
                        help="Use casefolding (aggresive lowercasing) for text before evaluation. Not used by default.")
    return parser.parse_args()

def main():

    args = initialize_argparse()

    predictions = read_jsonl(args.predictions)
    references = read_jsonl(args.references)
    assert len(predictions) == len(references)

    scores = calculate_metrics(predictions=predictions, references=references, metric=args.metrics, lowercase=args.lower)
    print(f"Number of examples: {len(predictions)}")
    print(scores)

if __name__ == "__main__":
    main()

    