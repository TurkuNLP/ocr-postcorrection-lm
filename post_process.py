from Bio.Align import PairwiseAligner
import unicodedata
import json


def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def align(input, prediction):

    # Set gap scores
    # I tried to play with these but these didn't seem to do much
    #aligner.open_gap_score = -1
    #aligner.extend_gap_score = -0.1

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    alignments = aligner.align(input, prediction)
    alignment = alignments[0]
    return alignment


def prune(prediction, alignment):
    prediction_alignment = alignment.aligned[1]
    start_idx = prediction_alignment[0][0]
    end_idx = prediction_alignment[-1][-1]
    pruned_prediction = prediction[start_idx:end_idx]
    return pruned_prediction

def main(args):

    with open(args.input_file, "rt", encoding="utf-8") as f, open(args.output_file, "wt", encoding="utf-8") as f_out:
        for line in f:
            example = json.loads(line)
            orig_input = example["input"]
            prediction = example["output"]

            # (1) prune extra (unaligned) characters from beginning and end
            alignment = align(normalize_unicode(orig_input).lower(), normalize_unicode(prediction).lower()) # this normalization does not delete anything so char_idx keeps unchanged
            pruned_prediction = prune(prediction, alignment)

            # (2) TODO: delete predictions that are too short?
            #if len(pruned_prediction) < len(orig_input)*0.9:
                # prediction too short, revert to original
            #    print("Reverting to original prediction.")
            #    pruned_prediction = orig_input

            d = {"input": orig_input, "output": pruned_prediction}
            print(json.dumps(d), file=f_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    main(args)