from Bio.Align import PairwiseAligner

from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage
import ollama

from collections import defaultdict
from datetime import datetime

import re
from math import ceil 
import numpy as np 
import traceback 

import logging
logging.getLogger('transformers').setLevel(logging.ERROR) ## in order to disable """A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'`  when initializing the tokenizer.""", which in our case does not hold.

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) #print stuff in red 

def correct(text, model=None, tokenizer=None, ollama=False):
    window_size = 300 ###TODO replace window size with best number
    prompts, texts, reference_numbers = prepare_text_input(text, tokenizer, slicing = True, window_size = window_size) 
    print(texts)
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
        alignment, alignment_array = align(normalized_input["text"], normalized_output["text"])

        print(alignment[0])
        alignment_string = craft_alignment_string(alignment)
        print(alignment_string)
        print(alignment[1])
        start, end = new_extract_aligned_text(normalized_output, alignment, alignment_array)
        
        aligned_output = model_output[start:end]
        generated_outputs.append(aligned_output)
    print(len(generated_outputs), "outputs have been generated.\nMerging...")
    prRed("aligned output1")
    print(generated_outputs[0])
    prRed("aligned output2")
    print(generated_outputs[1])
    prRed("aligned output3")
    print(generated_outputs[2])
    
    shen_time = datetime.now()
    
    corrected_texts = generated_outputs

    while len(corrected_texts)!=1:
        for i in range((3-len(corrected_texts))%3):
            corrected_texts.append("")
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
    return corrected_texts[0]
    

def run_ollama(model, prompts, temperature):
    #ollama = ChatOllama(model=model, temperature=0.8)
    #prompts = [ [HumanMessage(content=prompt)] for prompt in prompts]
    response = ollama.chat(model='llama3.1', messages=[
                          {
                            'role': 'user',
                            'content': prompts[O],
                          },
                        ],
                        options = {
                          'temperature': temperature,
                        })
    print("generating...")

    return [ response['message']['content'] ]


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
    overlap = int(window_size*0.3)
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
        print(n, "tokens")
        mapping = tokenized_text["offset_mapping"]
        number_of_slices = max(1, ceil(n/window_size))
        
        #if number_of_slices > 1 :
         #   number_of_slices-=1       only for window size evaluation
        
        for i in range(number_of_slices):
            print('i', i)
            try:
                text = input[int(mapping[0][i*(window_size-overlap)][0]):int(mapping[0][i*(window_size-overlap)+window_size][0])]
            except:
                text = input[int(mapping[0][i*(window_size-overlap)][0]):int(mapping[0][-1][1])]
                
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
def new_extract_aligned_text(normalized_text, alignment, alignment_array, reversed=False):
    input_text = normalized_text["text"]
    map = normalized_text["map"]
    aligned_text = alignment[1]
    shape = alignment_array.shape

    if not reversed:
        word_start, word_end = alignment_array[1, 0][0], alignment_array[1, shape[1]-1][1]  
    else:
        word_start, word_end = alignment_array[0, 0][0], alignment_array[0, shape[1]-1][1] 

    start, end = map["words"][word_start], map["words"][word_end-1]+1

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
    from transformers import AutoTokenizer
    import subprocess

    subprocess.call("ollama.sh")
    cache_dir = '/scratch/project_2005072/ocr_correction/.cache'  # path to the cache dir where the models are
    with open("huggingface_key.txt", "r") as f:
        access_token = f.readline()
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", token=access_token)
    tokenizer.pad_token = tokenizer.eos_token  
    
    print("starting correction")
    
    test = astral.correct("""
    When, .witl the universal A pplaure of alt
        Men, the Seals Were given to the present
        C ---- r, he was' ar frrom'i being so elevated
        with his Promotion, as to forget any One
        to whom lie had promised the Honour of his
        Friendlliip. An Opportunity soon offered
        in Favour of Dr R----, the B---k ox G----r
        became vacant, .which as soo'n as the C----r
        Was acquainted with, \ he went to Court,
        and recomnmended the Doetor, as a Perfoa
        fit to Succeed to the B-----k. The DoAos
        was then approved of, and a C---'E---
        order'd out accordingly. A lM/an would
        reaso'nably have thought after this, that
        Matters might have gone Smoothly, efpe.
        cially as it is partly the Right of every new
        C--- r, to' recommend a fit Person to his
        M .------, to' fill the See that shall becom'
        eax¢ant after ihs comiig to the Seals.
        
    However, contrary to all expectations, unforeseen difficulties arose. 
    The noblemen of the realm, who had their own candidates, 
    began to murmur and sow discord among the courtiers. 
    The C----r found himself in a perplexing position, 
    trying to uphold his honour while placating the influential 
    factions within the court. Despite the approval of the Doctor, 
    whispers of dissent grew louder, 
    and intrigues within the palace became more convoluted.
    
    he Doctor, unaware of the brewing storm, prepared himself diligently for his new responsibilities. 
    He was a man of learning and integrity, well-liked by his peers and admired by his students. 
    Yet, the political landscape was treacherous. Letters filled with veiled threats and promises 
    of support started circulating, each trying to sway the C----r's decision or undermine his authority.
    
    The C----r, unwavering in his commitment, sought the counsel of his closest advisors. 
    Among them was Sir W----, a man of seasoned wisdom and unwavering loyalty. 
    Sir W---- advised caution and suggested a strategy to strengthen the Doctor's position by garnering the support 
    of the influential church leaders. "A show of unity and strength," he said, 
    "will silence the naysayers and ensure the Doctor's smooth ascension.
    
    In his speech, the C----r extolled the virtues of the Doctor, 
    highlighting his contributions to the field of medicine and his unwavering commitment 
    to the welfare of the people. His words were met with applause, and for a moment, 
    it seemed as if the opposition had been quelled. The Doctor was officially installed 
    in his new position, and a grand celebration ensued.""",
                          
    model="llama3", tokenizer=tokenizer, ollama=True)
    
    print("\n\n\n")
    print("corrected text:\n")
    print(test)
    
    
