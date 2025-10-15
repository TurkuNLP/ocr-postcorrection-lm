import json
import tqdm
import re
import time
from textwrap import wrap

# url <class 'str'>
# text-orig <class 'list'>
# answer <class 'list'>
# post-processed-answer <class 'list'>
# removed-chunks <class 'list'>
# flagged <class 'str'>


def get_character_stats():

    path = "/scratch/project_2005072/cassandra/ecco-ocr-run-out-assessment/postprocessed.v2/"

    original_characters = 0
    corrected_characters = 0
    postprocessed_characters = 0

    for fnumber in range(200):
        with open(f"{path}/{fnumber}.jsonl", "rt", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc=f"Reading file {fnumber}"):
                book = json.loads(line)
                assert len(book["text-orig"]) == len(book["answer"]) == len(book["post-processed-answer"])
                for s in range(len(book["text-orig"])):
                    original_characters += len("".join(book["text-orig"][s].split()))
                    corrected_characters += len("".join(book["answer"][s].split()))
                    postprocessed_characters += len("".join(book["post-processed-answer"][s].split()))


    print(f"Original characters: {original_characters:,}")
    corrected_diff = ((corrected_characters - original_characters) / original_characters) * 100
    print(f"Corrected characters: {corrected_characters:,} ({corrected_diff:+.1f}% compared to original)")
    postprocessed_diff = ((postprocessed_characters - original_characters) / original_characters) * 100
    print(f"Postprocessed characters: {postprocessed_characters:,} ({postprocessed_diff:+.1f}% compared to original)")



def processed_documents():
    doc_ids = set()
    with open("../ecco-paper-data/aligned_ecco_books_compact.jsonl", "rt", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Reading already produced alignments"):
            book = json.loads(line)
            doc_ids.add(book["url"])
    print(f"Processed documents: {len(doc_ids)}")
    return doc_ids



def compact(text):
    text = text.replace("\n", "")
    text = text.replace(" ", "")
    text = text.replace("-", "")
    return text


def yield_alignments(fname):
    with open(fname, "rt", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            yield a["url"], a["correction-original"]

def get_edit_stats():

    ready_documents = processed_documents()

    alignment_file = "/scratch/project_2005072/cassandra/ecco-correction-aligned.jsonl"
    #alignment_file = "delme.jsonl"
    path = "/scratch/project_2005072/cassandra/ecco-ocr-run-out-assessment/postprocessed.v2/"

    output_fname = "../ecco-paper-data/aligned_ecco_books_compact.jsonl"
    output_file = open(output_fname, "at", encoding="utf-8")

    # read alignments from file
    book_alignments = yield_alignments(alignment_file)

    # iterate over books
    counter = 0
    for fnumber in range(200):
        with open(f"{path}/{fnumber}.jsonl", "rt", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc=f"Reading file {fnumber}"):
                book = json.loads(line)
                if book["url"] == "./all_files.txt": # continue without proceeding in alignments
                    print(f"Skipping metadata file {book['url']}")
                    continue
                url, alignments = next(book_alignments)
                if book["url"] != url:
                    print(f"Book {book['url']} does not match alignment {url}.")
                    exit()
                if book["url"] in ready_documents:
                    continue


                original_text = " ".join(book["text-orig"])
                corrected_text = " ".join(book["post-processed-answer"])
                
                
                final_original = ""
                final_corrected = ""
                final_alignments = ""

                # iterate over aligned parts, everything between these are unaligned (either deletions or additions)
                original_char_index = 0
                corrected_char_index = 0
                for corrected_index, original_index in zip(alignments[0], alignments[1]):
                    corrected_start, corrected_end = corrected_index
                    original_start, original_end = original_index

                    # collect deletions/insertions from unaligned parts until this alignment
                    # everything remaining in original_text is a deletion
                    if original_char_index < original_start:
                        chars = compact(original_text[original_char_index:original_start])
                        for c in chars:
                            final_original += c
                            final_corrected += " " # space for unaligned part
                            final_alignments += "-" # deletion
                        original_char_index = original_start # TODO
                    # everything remaining in corrected_text is an insertion
                    if corrected_char_index < corrected_start:
                        chars = compact(corrected_text[corrected_char_index:corrected_start])
                        for c in chars:
                            final_corrected += c
                            final_original += " " # space for unaligned part
                            final_alignments += "+" # insertion
                        corrected_char_index = corrected_start # TODO

                    # collect aligned parts
                    a_original = compact(original_text[original_start:original_end])
                    a_corrected = compact(corrected_text[corrected_start:corrected_end])
                    assert len(a_original) == len(a_corrected), f"Original: {a_original} ({len(a_original)}), Corrected: {a_corrected} ({len(a_corrected)})"
                    for c1, c2 in zip(a_original, a_corrected):
                        final_original += c1
                        final_corrected += c2
                        if c1 == c2:
                            final_alignments += "|" # space for aligned part
                        else:
                            final_alignments += "." # space for aligned part

                    original_char_index = original_end
                    corrected_char_index = corrected_end
                    
                # remaining characters after last alignment
                if len(original_text[original_char_index:]) > 0:
                    remaining_chars = compact(original_text[original_char_index:])
                    for c in remaining_chars:
                        final_original += c
                        final_corrected += " " # space for unaligned part
                        final_alignments += "-" # deletion
                if len(corrected_text[corrected_char_index:]) > 0:
                    remaining_chars = compact(corrected_text[corrected_char_index:])
                    for c in remaining_chars:
                        final_corrected += c
                        final_original += " " # space for unaligned part
                        final_alignments += "+" # insertion
                assert len(final_original) == len(final_corrected) == len(final_alignments)



                # print to jsonl file
                print(json.dumps({"url": url, "original": final_original, "corrected": final_corrected, "alignments": final_alignments}, ensure_ascii=False), file=output_file, flush=True)
                counter += 1 
                
                # debug prints
                if counter % 200 == 0: # for every 200th book, print every 100th segment
                    s = 0
                    for i in range(0, len(final_original), 100):
                        if s % 100 == 0:
                            print()
                            print(final_original[i:i+100])
                            print(final_corrected[i:i+100])
                            print(final_alignments[i:i+100])
                            print()
                        s += 1


                    

    output_file.close()
                


# get_character_stats()

get_edit_stats()