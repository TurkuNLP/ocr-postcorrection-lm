from evaluate import load
import unicodedata

wer_metric = load("wer")
cer_metric = load("cer")
character_metric = load('character', 'wer')

def normalize(text):
    return unicodedata.normalize("NFKC", text)

def calculate_metrics(*, predictions, references):
    references = [normalize(r) for r in references]
    predictions = [normalize(p) for p in predictions]
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    character_score = character_metric.compute(predictions=predictions, references=references)["cer_score"]
    return {"wer": wer_score, "cer": cer_score, "character_wer": character_score}

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="The predictions file (prediction=output).")
    parser.add_argument("--references", type=str, required=True, help="The references file (reference=output).")
    args = parser.parse_args()

    predictions = []
    with open(args.predictions, "rt", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            predictions.append(example["output"])

    references = []
    with open(args.references, "rt", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            references.append(example["output"])

    assert len(predictions) == len(references)
    print(f"Number of examples: {len(predictions)}")
    metrics = calculate_metrics(predictions=predictions, references=references)
    print(metrics)