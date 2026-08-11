# code to produce fully character-by-character aligned TCP books with original, corrected and reference texts.
# example: 
# original:  "the quik- bromn box jumkk- ov-r the aa lazy dog."
# corrected: "The quick brown box jumped over the -- lazy dog."
# reference: "The quick brown fox jumps- over the -- lazy dog."


import json
import tqdm
import re
import time
from textwrap import wrap
import time
import ast
import argparse
import sys
from collections import Counter

# url <class 'str'>
# text-orig <class 'list'>
# answer <class 'list'>
# post-processed-answer <class 'list'>
# removed-chunks <class 'list'>
# flagged <class 'str'>


def parse_large_alignment_string(s):
    s = re.sub(r'np\.int64\((\d+)\)', r'\1', s)
    split_match = re.search(r'\]\s*,\s*\[', s)
    if not split_match: return [[], []]
    part1 = s[:split_match.start()]
    part2 = s[split_match.end():]
    def extract(sub):
        raw_pairs = re.findall(r'(\d+)\s*,\s*(\d+)', sub)
        return [[int(p[0]), int(p[1])] for p in raw_pairs]
    return [extract(part1), extract(part2)]



def compact(text):
    text = text.replace("\n", "")
    text = text.replace(" ", "")
    text = text.replace("-", "")
    #text = text.replace("\u0085", "") # find Suchgaftlynoyfcofyronch (book 0373500101)
    return text


def read_alignments(from_direction, alignment_file):
    import zipfile
    alignments = {}
    if alignment_file.endswith(".zip"):
        with zipfile.ZipFile(alignment_file, "r") as f: # assumes zipfile, TODO: support jsonl file
            for name in tqdm.tqdm(f.namelist(), desc="Reading alignments"):
                with f.open(name, "r") as af:
                    # Read content as text
                    content = af.read().decode('utf-8')
                    # Try JSON first, then Python literal (for tuples, etc.)
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # If JSON fails, try Python literal evaluation (handles tuples, etc.)
                        data = ast.literal_eval(content)    
                    alignments[name] = data
    elif alignment_file.endswith(".jsonl"):
        with open(alignment_file, "rt", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc="Reading alignments"):
                book = json.loads(line)
                if f"{from_direction}-reference" in book:
                    data = book[f"{from_direction}-reference"]
                    if not isinstance(data, list):
                        data = parse_large_alignment_string(data)
                    alignments[book["url"]] = data
    else:
        raise ValueError(f"Invalid alignment file: {alignment_file}")
    return alignments


def align(from_direction, alignment_file):

    # 3 books excluded
    ner_books = ['0161700112', '0671103100', '0134304400', '0081903100', '0157502000', '0101801800', '1034200202', '0653500600', '0406001100', '0055100300', '1083401300', '1296001000', '0774900104', '0832500700', '0238500502', '0185702900', '1087800800', '0793800202', '0023100303', '0661600502', '0257700103', '0149800300', '0199100802', '1083800902', '0466800700', '0048202100', '0593400202', '0568403600', '0238500401', '0081902300', '1034200203', '0145900100', '1094700700', '1112200303', '0345900400', '0195400701', '0223501300', '0651901700', '0044000700', '0101702400', '1217900600', '1017801400', '0096302700', '0143500600', '0148502702', '0376600104', '0673700400', '0765801100', '0095100801', '0468800100', '0147503600', '0888000300', '1211504000', '0645300402', '0170200302', '0057401500', '0251102300', '0388401400', '0387600800', '0313900300', '0578804200', '0266900800', '0149503300', '0600700105', '0425900900', '0427400400', '1210500900', '1256001100', '0744200100', '0497900300', '1118600200', '0454100800', '0135100200', '0122702900', '0038501200', '0413000400', '0622700303', '0712300300', '1294600800', '0167001000', '0464004000', '0622600706', '0297001300', '0058702200', '0667301600', '1226600300', '0398000400', '0590900600', '0098301900', '0478900600', '0238600103', '0165000300', '0649001300', '0373500101', '1182801700', '1235100200', '0341602600']

    # alignments (numerical)
    book_alignments = read_alignments(from_direction, alignment_file)
    print(f"Read {len(book_alignments)} alignments")

    # text data
    tcp_books = "/scratch/project_2005072/jenna/git_checkout/ocr-postcorrection-lm/ecco-paper-data/final_ecco_tcp_joined_books.jsonl"


    output_fname = "../ecco-paper-data/delme2.jsonl"
    output_file = open(output_fname, "wt", encoding="utf-8")

    # iterate over books
    counter = 0
    statistics = Counter()
    failed_books = []
    failed_alignments = []
    bad_alignments = []
    fail = False
    with open(tcp_books, "rt", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Reading TCP books"):
            fail = False
            book = json.loads(line)
            url = book["book_id"]

            if url not in ner_books:
                continue

            original_text = book["ecco input"]
            corrected_text = book["ecco corrected postprocessed"]
            reference_text = book["tcp reference"]



            if from_direction == "original":
                from_text = original_text
            elif from_direction == "correction":
                from_text = corrected_text
            else:
                raise ValueError(f"Invalid from-direction: {from_direction}")

            final_from = ""
            final_reference = ""
            final_alignments = ""

            alignment = book_alignments[url]

            # iterate over aligned parts, everything between these are unaligned (either deletions or additions)
            from_char_index = 0
            reference_char_index = 0

            aligned_characters = 0
            for from_index, reference_index in zip(alignment[0], alignment[1]):
                from_start, from_end = from_index
                reference_start, reference_end = reference_index

                #assert reference_end-reference_start == from_end-from_start, f"Reference alignment length: {reference_end-reference_start} != From alignment length:{from_end-from_start}"

                # collect deletions/insertions from unaligned parts until this alignment
                # everything remaining in original_text is a deletion
                if reference_char_index < reference_start:
                    chars = compact(reference_text[reference_char_index:reference_start])
                    #print("Extra in reference:", repr(chars))
                    for c in chars:
                        final_reference += c
                        final_from += " " # space for unaligned part
                        final_alignments += "-" # deletion
                    reference_char_index = reference_start # TODO
                # everything remaining in from_text is an insertion
                if from_char_index < from_start:
                    chars = compact(from_text[from_char_index:from_start])
                    #print("Extra in from:", repr(chars))
                    for c in chars:
                        final_from += c
                        final_reference += " " # space for unaligned part
                        final_alignments += "+" # insertion
                    from_char_index = from_start # TODO

                # collect aligned parts
                #a_reference = compact(reference_text[reference_start:reference_end])
                #a_from = compact(from_text[from_start:from_end])

                a_reference = compact(reference_text[reference_start:reference_end])
                a_from = compact(from_text[from_start:from_end])
                


                #print("\nR:",repr(a_reference))
                #print(f"F:",repr(a_from))
                    
                if len(a_reference) != len(a_from):
                    failed_books.append(url)
                    print(f"Reference: {len(a_reference)}, From: {len(a_from)} \nR: {a_reference}\nF: {a_from}")
                    fail = True
                    break
                aligned_characters += len(a_reference) + len(a_from)

                assert len(a_reference) == len(a_from), f"Reference: {len(a_reference)}, From: {len(a_from)} \nR: {a_reference}\nF: {a_from}"
                for c1, c2 in zip(a_reference, a_from):
                    final_reference += c1
                    final_from += c2
                    if c1 == c2:
                        final_alignments += "|" # space for aligned part
                    else:
                        final_alignments += "." # space for aligned part



                reference_char_index = reference_end
                from_char_index = from_end

            if fail:
                continue

            # sanity check after final alignments
            if aligned_characters / (len(compact(reference_text)) + len(compact(from_text))) < 0.5:
                failed_alignments.append(url)
            if (len(reference_text[reference_char_index:]) + len(from_text[from_char_index:]) > 200):
            #    print("Book:", url)
            #    print(f"Aligned characters: {aligned_character} / ({len(reference_text) + len(from_text)}) {aligned_characters/len(reference_text)+len(from_text)*100:.2f}%")
                
                if final_alignments[-200:].count("|") < 50:
                    print("Book:", url)
                    print(f"Aligned characters: {aligned_characters / (len(compact(reference_text)) + len(compact(from_text)))*100:.2f}%")
                    print(f"Unaligned characters at the end: {len(reference_text[reference_char_index:])} + {len(from_text[from_char_index:])} of total lengths {len(reference_text)} + {len(from_text)}")
                    print(f"Suspicious final alignment block:", )
                    print(final_reference[-200:])
                    print(final_from[-200:])
                    print(final_alignments[-200:])
                    print()
                    bad_alignments.append(url)
            #    suspicious_books.append(url)

            #else: # for loop run succesfully --> process rest, otherwise do nothing

            # remaining characters after last alignment
            if len(reference_text[reference_char_index:]) > 0:
                remaining_chars = compact(reference_text[reference_char_index:])
                for c in remaining_chars:
                    final_reference += c
                    final_from += " " # space for unaligned part
                    final_alignments += "-" # deletion
            if len(from_text[from_char_index:]) > 0:
                remaining_chars = compact(from_text[from_char_index:])
                for c in remaining_chars:
                    final_from += c
                    final_reference += " " # space for unaligned part
                    final_alignments += "+" # insertion
            

            assert len(final_reference) == len(final_from) == len(final_alignments), "Output length mismatch"
            assert compact(reference_text) == compact(final_reference), "Reference text mismatch"
            assert compact(from_text) == compact(final_from), f"{from_direction} text mismatch"

            #print("original-reference.jsonl", file=sys.stderr)
            #print("C:",final_from, file=sys.stderr)
            #print("A:", final_alignments, file=sys.stderr)
            #print("R:", final_reference , file=sys.stderr)
            #exit()

            # print to jsonl file
            print(json.dumps({"url": url, "reference": final_reference, f"{from_direction}": final_from, "alignments": final_alignments}, ensure_ascii=False), file=output_file, flush=True)
            counter += 1

            statistics.update(list(final_alignments))

            

    #print("\nStatistics:")
    #for c, count in statistics.most_common():
    #    print(c, count, f"{count/statistics.total()*100:.2f}%")

    print("\nfailed books:", len(failed_books), failed_books)
    print("\nfailed alignments:", len(failed_alignments), failed_alignments)
    print("\nbad alignments:", len(bad_alignments), bad_alignments)
    print()
    print()

    output_file.close()
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-direction", choices=["original", "correction"], help="Alignment from [this] to reference.")
    parser.add_argument("--alignment_file", help="Alignment file, must match from-direction.")
    args = parser.parse_args()
    

    #possible alignment files:
    # /scratch/project_2005072/cassandra/
    # original-reference2.zip
    # correction-reference.jsonl


    to_direction = "reference"

    align(args.from_direction, args.alignment_file)