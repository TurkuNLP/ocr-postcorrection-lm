from evaluate import load
import unicodedata
import argparse
import json
import numpy as np


def normalize(text: str, lowercase=False, modernize=False, remove_newlines=False):
    if remove_newlines:
        text = (text.replace("-\n", "-")).replace("\n", " ")

    if modernize:
        text = text.replace("w", "v").replace("W", "V")

    if lowercase:
        return unicodedata.normalize("NFKC", text.casefold())

    return unicodedata.normalize("NFKC", text)

def calculate_metrics(*, predictions, references, originals=None, metric="cer", lowercase=False, modernize=False, do_normalize=True, remove_newlines=False):

    if do_normalize: 
        references = [normalize(r, lowercase, modernize, remove_newlines) for r in references]
        predictions = [normalize(p, lowercase, modernize, remove_newlines) for p in predictions]
        if originals:
            originals = [normalize(o, lowercase, modernize, remove_newlines) for o in originals]

    scores = {"micro": {}, "mean": {}, "median": {}}
    all_document_scores = {}

    # init metrics
    metrics = []
    if metric == "cer":
        metrics.append(("cer", load("cer")))
    elif metric == "wer":
        metrics.append(("wer", load("wer")))
    elif metric == "character":
        metrics.append(("character", load("character")))
    elif metric == "all":
        metrics = [("cer", load("cer")), ("wer", load("wer"))]
        #metrics = [("cer", load("cer")), ("wer", load("wer")), ("character", load("character"))]
    else:
        assert False, "Unknown metric."

    for metric_name, metric_obj in metrics:
        scores["micro"][metric_name] = metric_obj.compute(predictions=predictions, references=references)
        document_scores = []
        for p, r in zip(predictions, references):
            s = metric_obj.compute(predictions=[p], references=[r])
            if metric_name == "character":
                s = s["cer_score"]
            document_scores.append(s)
        scores["mean"][metric_name] = np.mean(document_scores)
        scores["median"][metric_name] = np.median(document_scores)
        all_document_scores[metric_name] = document_scores
    
        scores["individual"] = all_document_scores

    if originals:
        scores["original"] = {}
        # calculate improvements
        scores["improvement"] = {}
        # weights for each document, calculated by using the ocr_doc / total_ocr_docs
        total_ocr_chars = sum([len(p) for p in predictions])
        weights = [len(p)/total_ocr_chars for p in predictions]
        for metric_name, metric_obj in metrics:
            scores["improvement"][metric_name] = {}
            orig_scores = []
            for o, r in zip(originals, references):
                s = metric_obj.compute(predictions=[o], references=[r])
                if metric_name == "character":
                    s = s["cer_score"]
                orig_scores.append(s)
            improvements = []
            pred_scores = all_document_scores[metric_name]
            for pred_d, orig_d in zip(pred_scores, orig_scores):
                if orig_d == 0:
                    impv = pred_d
                else:
                    impv = (orig_d - pred_d) / orig_d
                #impv = min(max(impv, -1), 1) # cut to -1, 1
                improvements.append(impv)
            scores["improvement"][metric_name]["mean"] = np.mean(improvements)
            scores["improvement"][metric_name]["median"] = np.median(improvements)
            scores["improvement"][metric_name]["weighted average"] = np.average(improvements, weights=weights)
            scores["improvement"][metric_name]["individual"] = improvements
            scores["original"][metric_name] = orig_scores
    return scores

def evaluate(*, p_args, r_args, metric='cer', lowercase=False, modernize=False):
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
    metrics = calculate_metrics(predictions=predictions, references=references, metric=metric, lowercase=lowercase, modernize=modernize)

    return metrics


def read_jsonl(args):

    if len(args) == 2:
        key = args[1]
    else: key = "output"

    examples = []
    with open(args[0], "rt", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            if "originals" in example:
                example = example["originals"]
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
    parser.add_argument("-o",
                        "--originals",
                        type=str,
                        required=False,
                        nargs="+",
                        help="The name of / path to the originals file. REQUIRED: if originals is given, the key is required.")
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
    parser.add_argument("--modernize",
                        default=False,
                        action='store_true',
                        help="Modernize the text before evaluation (replace w with v).")
    return parser.parse_args()

def main():

    args = initialize_argparse()

    predictions = read_jsonl(args.predictions)
    references = read_jsonl(args.references)
    assert len(predictions) == len(references)

    if args.originals:
        originals = read_jsonl(args.originals)
        assert len(predictions) == len(originals)
    else:
        originals = None

    print(f"Unicode NFKC normalization: True")
    print(f"Lowercase: {args.lower}")
    print(f"Modernize (w --> v): {args.modernize}")

    scores = calculate_metrics(predictions=predictions, references=references, originals=originals, metric=args.metrics, lowercase=args.lower, modernize=args.modernize)
    print(f"Number of examples: {len(predictions)}")
    print(scores)

    if originals:
        print("\nScores for original OCR (original vs. reference):")
        scores = calculate_metrics(predictions=originals, references=references, originals=None, metric=args.metrics, lowercase=args.lower, modernize=args.modernize)
        print(scores)

if __name__ == "__main__":
    main()

    