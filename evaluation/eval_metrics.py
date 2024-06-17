from evaluate import load
import unicodedata
import argparse
import json

wer_metric = load("wer")
cer_metric = load("cer")
character_metric = load("character")

def normalize(text):
    return unicodedata.normalize("NFKC", text)

def calculate_metrics(*, predictions, references, metric='cer'):

    references = [normalize(r) for r in references]
    predictions = [normalize(p) for p in predictions]

    scores = {}

    if metric == "cer" or metric == "all":
        scores["cer"] = cer_metric.compute(predictions=predictions, references=references)
    
    if metric == "wer" or metric == "all":
        scores["wer"] = wer_metric.compute(predictions=predictions, references=references)
    
    if metric == "character" or metric == "all":
        scores["character"] = character_metric.compute(predictions=predictions, references=references)["cer_score"]

    return scores

def evaluate(*, p_args, r_args, metric='cer'):
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
    metrics = calculate_metrics(predictions=predictions, references=references, metric=metric)

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
    return parser.parse_args()

def main():

    args = initialize_argparse()

    predictions = read_jsonl(args.predictions)
    references = read_jsonl(args.references)
    assert len(predictions) == len(references)

    metrics = calculate_metrics(predictions=predictions, references=references, metric=args.metrics)
    print(f"Number of examples: {len(predictions)}")
    print(metrics)

if __name__ == "__main__":
    main()

    