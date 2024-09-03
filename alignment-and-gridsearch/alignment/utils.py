from Bio.Align import PairwiseAligner
import ollama
from collections import defaultdict

import logging
logging.getLogger('transformers').setLevel(logging.ERROR) ## in order to disable """A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'`  when initializing the tokenizer.""", which in our case does not hold.

def run_ollama(model, prompts, model_options, verbose=False):
    response = ollama.chat(model=model, messages=[
                          {
                            'role': 'user',
                            'content': prompts[0],
                          },
                        ],
                        options = model_options)
    if verbose:
        print("generating...")
        
    return [ response['message']['content'] ]

### BASE ALIGNMENT
def align(input_text, output_text):
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5 # trying out stuff
    alignments = aligner.align(input_text, output_text)
    alignment = alignments[0]
    alignment = [alignment[0].replace("-", " "), alignment[1].replace("-", " ")]
    return alignment, alignments[0].aligned


### STORE THE INDICES OF FORMATTING CHARACTERS
def mapping_format(input_text):
    spaces = []
    new_lines = []
    words = []
    dashes = []

    for index, char in enumerate(input_text):
        if char==" ":
            spaces.append(index)
        elif char=="\n":
            new_lines.append(index)
        elif char=="-":
            dashes.append(index)
        else:
            words.append(index)
                   
    map = {
        "spaces": spaces,
        "new_lines": new_lines,
        "words": words,
        "dashes": dashes
    }
    
    return map


def suppress_format(input_text):   # delete spaces and newlines and provide a mapping of the original format for reversibility
    map = mapping_format(input_text)
    text = input_text.replace("\n", "").replace(" ", "").replace("-", "")
    return {"map": map, "text": text}


def revert_to_original_format(input_text, map): # does the opposite of the function above
    length = len(map["spaces"]) + len(map["new_lines"]) + len(map["dashes"]) + len(map["words"])
    text = [None for i in range(length)]
    
    for elt in map["spaces"]:
        text[elt]= " "
    for elt in map["new_lines"]:
        text[elt] = "\n"
    for elt in map["dashes"]:
        text[elt] = "-"
    for idx, elt in enumerate(map["words"]):
        text[elt] = input_text[idx]

    original_text = ''.join(text)
    
    return original_text

### EXTRACTION OF ORIGINAL TEXT 
def new_extract_aligned_text(normalized_text, alignment, alignment_array, reversed=False):  #TODO - rename
    map = normalized_text["map"]
    shape = alignment_array.shape
    print(reversed)
    if not reversed:
        
        word_start, word_end = alignment_array[1, 0][0], alignment_array[1, shape[1]-1][1]  
    else:
        word_start, word_end = alignment_array[0, 0][0], alignment_array[0, shape[1]-1][1] 

    start, end = map["words"][word_start], map["words"][word_end-1]+1

    return start, end 

### MANUAL CRAFT OF ALIGNMENT STRING 
def craft_alignment_string(alignment):   
    aligned_sequence1 = alignment[0]
    aligned_sequence2 = alignment[1]
    
    alignment_string=""
    
    for idx, char in enumerate(aligned_sequence1):
        if char==aligned_sequence2[idx]:
            alignment_string+="|"
        elif char==" " or aligned_sequence2[idx]==" ":
            alignment_string+="-"
        else:
            alignment_string+="."

    return alignment_string