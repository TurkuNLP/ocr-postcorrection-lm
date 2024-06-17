from Bio.Align import PairwiseAligner

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

import difflib
from collections import defaultdict
from datetime import datetime

import re

import logging
logging.getLogger('transformers').setLevel(logging.ERROR) ## in order to disable """A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.""", which in our case does not hold.


def correct(text, model=None, tokenizer=None, ollama=False):
    window_size = 300 ###TODO replace window size with best number
    prompts, texts, reference_numbers = prepare_text_input(text, tokenizer, slicing = True, window_size = window_size) 
    generated_outputs = []

    for index, prompt in enumerate(prompts):
        gen_time = datetime.now()
        if ollama:
            model_output = run_ollama(model, [prompt])[0]
        else:
            model_output = run_lm(model, tokenizer, prompt, window_size)
        gen_time = round((datetime.now()-gen_time).total_seconds(), 1)
        print("generation time = ", gen_time, "seconds for", len(model_output), "characters")
        normalized_output = suppress_format(model_output)
        normalized_input = suppress_format(texts[index])
        alignment = align(normalized_input["text"], normalized_output["text"])

        print(alignment[0])
        alignment_string = craft_alignment_string(alignment)
        print(alignment_string)
        print(alignment[1])
        start, end = extract_aligned_text(normalized_output, alignment[1], alignment_string)
        
        aligned_output = model_output[start:end]
        generated_outputs.append(aligned_output)
    print(len(generated_outputs), "outputs have been generated.\nMerging...")
    print(generated_outputs)
    shen_time = datetime.now()
    for i in range(3-len(generated_outputs)%3):
        generated_outputs.append("")

    corrected_texts = generated_outputs

    while len(corrected_texts)!=1:
        temp = []
        for i in range(len(corrected_texts)//3):
            
            str1 = corrected_texts[3*i]
            str2 = corrected_texts[(3*i)+1]
            str3 = corrected_texts[(3*i)+2]
    
            corrected_text = weighted_merge(str1, str2, str3)
            temp.append(corrected_text)
        corrected_texts = temp
    shen_time = round((datetime.now()-shen_time).total_seconds(), 1)
    print("processes other than generations took ", shen_time, "seconds.")
    return corrected_texts
    

def run_ollama(model, prompts):
    ollama = ChatOllama(model=model, temperature=0.8)
    prompts = [ [HumanMessage(content=prompt)] for prompt in prompts]
    print("started generating")
    results = ollama.generate(prompts)
    print("done with generation")
    return [ result[0].text for result in results.generations ]


def get_prompt_size(tokenizer):  # for now this is only used to get to know how long the prompt is in terms of token length to not cut it when slicing the text
    device="cuda"
    messages = [
        {"role": "system", "content": "You are a useful assistant"},
        {"role": "user", "content": f""" Correct the OCR errors in the text below. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n"""},
    ]
    encoded = tokenizer.apply_chat_template(messages, return_dict=True, return_tensors="pt").to(device)
    prompt_token_size = encoded["input_ids"][0].size()[0]-1  # -1 to exclude the eot token
                                                                                 
    return prompt_token_size 
    

def prepare_text_input(input, tokenizer, slicing=False, window_size=100):
    device="cuda"
    if not slicing:
        text = example["input"]
        prompt = f""" Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text}"""
        
        return [prompt], [text], [example["reference_number"]]
        
    if slicing:        
        prompts=[]
        texts=[]
        reference_numbers=[]
        tokenized_text = tokenizer(input,return_offsets_mapping=True, return_tensors="pt").to(device)

        n = tokenized_text["input_ids"].size()[1]
        mapping = tokenized_text["offset_mapping"]
        number_of_slices = max(1, n//window_size)
        
        #if number_of_slices > 1 :
         #   number_of_slices-=1       only for window size evaluation
        
        for i in range(number_of_slices):
            print('i', i)
            try:
                text = input[int(mapping[0][i*window_size][0]):int(mapping[0][(i+1)*window_size][0])]
            except:
                text = input[int(mapping[0][i*window_size][0]):int(mapping[0][-1][1])]
                
            prompt = f""" Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text}"""
            texts.append(text)
            prompts.append(prompt)
            #reference_numbers.append(example["reference_number"])

        return prompts, texts, reference_numbers


### run the llm to correct
def run_lm(model, tokenizer, prompt, window_size):
    device="cuda"
    messages = [
        {"role": "system", "content": "You are a useful assistant"},
        {"role": "user", "content": prompt},
    ]
    prompt_size = get_prompt_size(tokenizer)
    encoded = tokenizer.apply_chat_template(messages, return_dict=True,  return_tensors="pt", max_length = int(1.5*(prompt_size+window_size))).to(device)
    generated_ids = model.generate(**encoded, max_new_tokens=1.5*window_size, pad_token_id=tokenizer.eos_token_id, temperature=0.8, do_sample=True, num_beams=4, early_stopping=True)    ### TODO :  replace with optimal parameters
    decoded = tokenizer.batch_decode(generated_ids)
    model_output = decoded[0].split("<|end_header_id|>")[-1].strip("<|end_of_text|>")
    
    return model_output

### glueing-back test
def find_overlap(s1, s2):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    if match.size > 0:
        return s1[:match.a] + s2[match.b:], match.a, match.b
    else:
        return s1 + s2, len(s1), len(s2)

### weight function for the characters in the string
def weightify(x, length):
    y = x-length/2
    y = y**2
    y = -y
    y = y /(length/2)**2
    y += 1

    return y
    
### MERGING 3 STRINGS
def weighted_merge(str1, str2, str3):
    merged, a, b = find_overlap(str1, str2)  
    m = len(merged)
    merged, c, d = find_overlap(merged, str3)

    #print(merged)

    end1 = len(merged)-len(str1)
    start2 = a-b
    end2 = len(merged)-m
    start3 = len(merged)-len(str3)

    temp_str1 = str1 + "@"*end1
    temp_str2 = "@"*start2 + str2 + "@"*end2
    temp_str3 = "@"*start3 + str3
    
    #print(temp_str1)
    #print(temp_str2)
    #print(temp_str3)
    weights = [[ weightify(i, len(str1)) for i in range(len(str1))] + [0 for i in range(end1)], 
               [0 for i in range(start2)] + [ weightify(i, len(str2)) for i in range(len(str2))] + [0 for i in range(end2)], 
               [0 for i in range(start3)] + [ weightify(i, len(str3)) for i in range(len(str3))]]

    #print(weights[0])
    #print(weights[1])
    #print(weights[2])
    char_weights = defaultdict(lambda: defaultdict(int))

    for idx, char in enumerate(temp_str1):
        char_weights[idx][char] += weights[0][idx]
    for idx, char in enumerate(temp_str2):
        char_weights[idx][char] += weights[1][idx]
    for idx, char in enumerate(temp_str3):
        char_weights[idx][char] += weights[2][idx]
        
    final_string = []
    for i in range(len(merged)):
        #print(char_weights[i])
        if char_weights[i]:
            char, _ = max(char_weights[i].items(), key=lambda item: item[1]) # highest weight wins
            final_string.append(char)
        else:
            # If no weight information, just take the character from the merged string
            final_string.append(merged[i])

    return ''.join(final_string)

### MATCH SCORES (only used with pairwise2)
def match(l,r):
    if l=="s" and r=="ſ":
        return 1
    elif l=="ſ" and r=="s":
        return 1
    elif l==r:
        return 1
    else:
        return 0

### BASE ALIGNMENT
def align(input_text, output_text):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5 # trying out stuff
    alignments = aligner.align(input_text, output_text)
    alignment = alignments[0]
    alignment = [alignment[0].replace("-", " "), alignment[1].replace("-", " ")]
    return alignment

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
def extract_aligned_text(normalized_text, aligned_text, alignment_string):
    input_text = normalized_text["text"]
    map = normalized_text["map"]

    matches = list(re.finditer(r'[|.]+', alignment_string))
        
    last_match = max(matches, key=lambda m: m.end())
    first_match = min(matches, key=lambda m: m.start())
    word_start, word_end = first_match.start(), last_match.end()-1     # -1 here because the .end() is the first index not included, i.e. can can be out of bounds.
    new_word_start, new_word_end = word_start, word_end
    for index, char in enumerate(aligned_text[:word_end+1]):      #tricky for-loop to recalibrate indices according to the alignment gaps (" ")
        if char==" ":
            new_word_end-=1
            if index<word_start:
                new_word_start-=1
                
    start, end = map["words"][new_word_start], map["words"][new_word_end]

    return start, end 

### MANUAL CRAFT OF ALIGNMENT STRING 
# mostly created for alignment with smoothing window but also convenient for other operations
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


if __name__=="__main__":
    None
    
