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
                
            comp_last = len(normalized_ref["text"]) + 250
            
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

alignment_jsonl = "new-alignments.jsonl"

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

#"ecco_tcp_joined_books_v2.jsonl" #keys(['ecco input', 'ecco corrected', 'book_id', 'tcp input', 'tcp reference'])
#ner_books = ['0575401100', '0390900200', '0032300302', '0161700112', '0671103100', '0134304400', '0081903100', '0157502000', '0101801800', '1034200202', '0653500600', '0406001100', '0055100300', '1083401300', '1296001000', '0774900104', '0832500700', '0238500502', '0185702900', '1087800800', '0793800202', '0023100303', '0661600502', '0257700103', '0149800300', '0199100802', '1083800902', '0466800700', '0048202100', '0593400202', '0568403600', '0238500401', '0081902300', '1034200203', '0145900100', '1094700700', '1112200303', '0345900400', '0195400701', '0223501300', '0651901700', '0044000700', '0101702400', '1217900600', '1017801400', '0096302700', '0143500600', '0148502702', '0376600104', '0673700400', '0765801100', '0095100801', '0468800100', '0147503600', '0888000300', '1211504000', '0645300402', '0170200302', '0057401500', '0251102300', '0388401400', '0387600800', '0313900300', '0578804200', '0266900800', '0149503300', '0600700105', '0425900900', '0427400400', '1210500900', '1256001100', '0744200100', '0497900300', '1118600200', '0454100800', '0135100200', '0122702900', '0038501200', '0413000400', '0622700303', '0712300300', '1294600800', '0167001000', '0464004000', '0622600706', '0297001300', '0058702200', '0667301600', '1226600300', '0398000400', '0590900600', '0098301900', '0478900600', '0238600103', '0165000300', '0649001300', '0373500101', '1182801700', '1235100200', '0341602600']
#print("NER BOOKS:", len(ner_books))
count=0
with open(input_file, "r") as f: 
    for index_line, line in tqdm(enumerate(f)):
        
        #if index_line%200!=args.rank:
        #    continue
        if index_line < starting_line:
            continue
        book = json.loads(line)
        #if book["book_id"] not in ner_books:
        #    continue

        correction = book["ecco corrected postprocessed"]
        reference = book["tcp reference"]
        original = book["original ecco text"]#book["ecco input"]
        joined = " ".join(book["ecco input"])
        assert "".join(original.split()) == "".join(joined.split())
        
        
        #    print("O", repr(original[:100]))
        #    print("-", repr(" ".join(book["ecco input"])[:100]))
        #continue

        alignment_array_ref, flag1 = get_alignment_array(correction, reference, simple=False, book_name=book["book_id"])
        
        #alignment_array_orig, flag2 = get_alignment_array(correction, original, simple=False)

        original = split_text_into_segments(original)
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
                    }
            print(json.dumps(data, ensure_ascii=False), file=g, flush=True)

print("all is done :)")

    










        
