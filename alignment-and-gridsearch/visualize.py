# Read a jsonl file and visualize the alignment between the input and output text


from Bio.Align import PairwiseAligner
import unicodedata
import json


def normalize_for_visualization(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return unicodedata.normalize("NFKC", text)


def print_align(input_text, output_text):

    # normalize spacing, easier to visualize (Note that alignment does not match the original text after this!)
    input_text = normalize_for_visualization(input_text)
    output_text = normalize_for_visualization(output_text)

    # Set gap scores
    # I tried to play with these but these didn't seem to do much
    #aligner.open_gap_score = -1
    #aligner.extend_gap_score = -0.1

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    alignments = aligner.align(input_text, output_text)
    alignment = alignments[0]
    print(alignment)

#<class 'Bio.Align.Alignment'>
#['__add__', '__array__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__',
# '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__',
# '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
# '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_convert_sequence_string',
# '_format_generalized', '_format_pretty', '_format_unicode', '_get_row', '_get_row_col', '_get_row_cols_iterable',
# '_get_row_cols_slice', '_get_rows', '_get_rows_col', '_get_rows_cols_iterable', '_get_rows_cols_slice', 'aligned',
# 'coordinates', 'counts', 'format', 'frequencies', 'indices', 'infer_coordinates', 'inverse_indices', 'length',
# 'map', 'mapall', 'query', 'reverse_complement', 'score', 'sequences', 'shape', 'sort', 'substitutions', 'target']


def main(args):
    with open(args.input_file, "rt", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            input_text = example["input"]
            output_text = example["output"]
            print("***Top is input, bottom is output.***")
            print_align(input_text, output_text)
            if "orig_scores" in example:
                print("Orig score:", example["orig_scores"])
            if "gener_scores" in example:
                print("Gener score:", example["gener_scores"])
            print("\n\n\n\n\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    main(args)