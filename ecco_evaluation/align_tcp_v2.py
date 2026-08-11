import json 
import csv
import re
import unicodedata
import sys 
sys.path.append('/scratch/project_2005072/jenna/git_checkout/ocr-postcorrection-lm/alignment-and-gridsearch')
import alignment.utils as astral
from tqdm import tqdm 
import argparse
#import numpy as np

def suppress_format_with_mapping(text):
    normalized = astral.suppress_format(text)
    removed = len(text) - len(normalized["text"])
    return normalized, removed

def get_alignment_array(ref_list, list_compare, simple=True, book_name=""):
    flagged = False
    if simple:
        alignment_array = [[], []]
        added_len_ref = 0
        added_len_compare = 0
        for i in range(len(ref_list)):
            normalized_ref = astral.suppress_format(ref_list[i])
            normalized_compare = astral.suppress_format(list_compare[i])
            try:
                alignment, temp_alignment_array = astral.align(normalized_ref["text"], normalized_compare["text"])
            except (IndexError,ValueError) as e_:
                print("Warning, alignment failed (simple).", book_name)
                flagged = True
                temp_alignment_array = [[], []]
            for start, end in temp_alignment_array[0]:
                alignment_array[0].append((start + added_len_ref, end + added_len_ref))
            for start, end in temp_alignment_array[1]:
                alignment_array[1].append((start + added_len_compare, end + added_len_compare))
            added_len_ref += len(normalized_ref["text"])
            added_len_compare += len(normalized_compare["text"])
    else:
        alignment_array = [[], []]
        added_len_ref = 0
        added_len_compare = 0
        temp_alignment_array =[""]
        for i in range(len(ref_list)):
            normalized_ref = astral.suppress_format(ref_list[i])
            if len(normalized_ref["text"])==0:
                print("exc empty segment")
                continue
                
            comp_last = len(normalized_ref["text"]) + 10 
            
            normalized_compare = astral.suppress_format(list_compare[:len(ref_list[i])+3000])
            if len(normalized_compare["text"])==0:
                print("exc empty gt")
                continue
            last = normalized_compare["map"]["words"][min(comp_last, len(normalized_compare["map"]["words"])-1)]
            normalized_compare = astral.suppress_format(list_compare[:last])
            
            #print(len(temp_alignment_array[0]))
            #print(len(normalized_compare["text"])/len(normalized_ref["text"]))
                
            try:
                alignment, temp_alignment_array = astral.align(
                    normalized_ref["text"],
                    normalized_compare["text"]
                )
            except (IndexError,ValueError) as e_:
                print("Warning, alignment failed (non-simple).", book_name)
                flagged = True
                continue

            for start, end in temp_alignment_array[0]:
                alignment_array[0].append((start + added_len_ref, end + added_len_ref))
            for start, end in temp_alignment_array[1]:
                alignment_array[1].append((start + added_len_compare, end + added_len_compare))

            new_index=normalized_compare["map"]["words"][end-1]+1
            if len(temp_alignment_array[1])==0:
                print("empty temp")
                end=0

            if end==0:
                new_index=0

            last_len = added_len_compare
            added_len_ref += len(normalized_ref["text"])
            added_len_compare += end

            list_compare = list_compare[new_index:]

           
            
    return alignment_array, flagged

def split_text_into_segments(text, words_per_segment=300):
    words = text.split()
    segments = []
    for i in range(0, len(words), words_per_segment):
        segment = " ".join(words[i:i+words_per_segment])
        segments.append(segment)
    return segments
    
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int)
args = parser.parse_args()

alignment_jsonl = "failed-books-new-alignments.jsonl"

starting_line = 0 
try:   
    with open(alignment_jsonl, "r") as done:  
        for line in done: 
            starting_line+=1
    print("Found", starting_line, "alignments, starting from here.")
except: 
    print("Failed to read previous alignments, starting from line 0.")   
 

input_file = "/scratch/project_2005072/cassandra/ecco-ocr-run-out-assessment/ecco_tcp_joined_books_v3.jsonl"
#input_file = "/scratch/project_2005072/jenna/git_checkout/ocr-postcorrection-lm/ecco-paper-data/final_ecco_tcp_joined_books.jsonl"

failed_books = ["0902000401", "0453100201", "0394000300", "1002600700", "0695000400", "0681100800", "0631300200", "0652800200", "0548300102", "0442300101", "0075801700", "0293800400", "0110402500", "1289500202", "1289500201", "0679801400", "0516400600", "0500201900", "0181500200", "0792500103", "0356802000", "0548300101", "0294200100", "0462000200", "0570200100", "0019500400", "1169200700", "0140800500", "0699600701", "0357200700", "0792600104", "0653201700", "0023800201", "0661600502", "0435300100"]

count=0
with open(input_file, "r") as f: 
    for index_line, line in tqdm(enumerate(f)):
        
        #if index_line%200!=args.rank:
        #    continue
        if index_line < starting_line:
            continue
        book = json.loads(line)
        
        if book["book_id"] not in failed_books:
            continue

        correction = book["ecco corrected postprocessed"] # this is splitted into segments of about 300 subwords
        reference = book["tcp reference"] # full book (str)
        original = book["ecco input"] #book["original ecco text"] # this is splitted into segments of about 300 subwords

        # sanity
        #joined = " ".join(book["ecco input"])
        #assert "".join(original.split()) == "".join(joined.split())
        
        
        # calculate correction-reference alignment
        alignment_array_ref, flag1 = get_alignment_array(correction, reference, simple=False, book_name=book["book_id"])

        # TODO: we may want this as well? This is already in the full ecco alignment file, but maybe we want it here as well?
        #alignment_array_orig, flag2 = get_alignment_array(correction, original, simple=False)

        # calculate original-reference alignment
        
        #original = split_text_into_segments(original) # not needed, we already have a list
        alignment_array_ref2, flag3 = get_alignment_array(original, reference, simple=False, book_name=book["book_id"])
        


        #print(type(alignment_array_ref), len(alignment_array_ref))
        #print(alignment_array_ref)

        alignment_array_ref_nonnumpy = [[[int(i),int(j)] for (i,j) in alignment_array_ref[0]], [[int(i),int(j)] for (i,j) in alignment_array_ref[1]]]
        alignment_array_ref2_nonnumpy = [[[int(i),int(j)] for (i,j) in alignment_array_ref2[0]], [[int(i),int(j)] for (i,j) in alignment_array_ref2[1]]]
        
        with open(alignment_jsonl, "at") as g:
            data = { "url": book["book_id"],
                   "correction-reference": alignment_array_ref_nonnumpy,
                   #"correction-original": str(alignment_array_orig),
                   "original-reference": alignment_array_ref2_nonnumpy,
                   "flagged": True if flag1 or flag3 == True else False 
                    }
            print(json.dumps(data, ensure_ascii=False), file=g, flush=True)

print("all is done :)")

    










        
