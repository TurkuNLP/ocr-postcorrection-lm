print("importing...")

import json
import argparse
import transformers
import torch
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from random import randint
import gzip
import unicodedata
from eval_metrics import calculate_metrics
import optuna
from Bio.Align import PairwiseAligner

import re

print("transformers version: ", transformers.__version__)
print("bitsandybtes version: ", bitsandbytes.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/scratch/project_2000539/jenna/ocr-correction/by_page_dev_slim.jsonl.gz")
parser.add_argument("--output_file", type=str, help="Output file for results, not implemented yet") #TODO : clean output file with grdisearch results ?
parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name or path")
parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process")
parser.add_argument("--ntrials", type=int, default=5, help="Number of trials in the optune gridsearch")
parser.add_argument("--quantization", type=int, default=4, help="Number of bits for quantization (4, 8, 16)")
parser.add_argument("--batch_size", type=int, default=1, help="Set up a batch size for batched generation") # this code does not really support batching right now
args = parser.parse_args()

device = "cuda" # the device to load the model onto
cache_dir = '/scratch/project_2005072/ocr_correction/.cache'  # path to the cache dir where the models are

def get_prompt_info(tokenizer):  
    if args.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        instruction = """[INST] Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n""" 
    else:
        instruction = """ Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n"""
        
    prompt_token_size = tokenizer.encode(instruction, return_tensors="pt").size()[1] # get the size of the instruction text so we don't accidentally 
                                                                                     # cut it out when doing the sliding window
    return { "prompt_token_size" : prompt_token_size }


def prepare_text_input(example):
    text = example["input"]
    if args.model =="mistralai/Mixtral-8x7B-Instruct-v0.1":
        prompt = f"""[INST] Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text} [/INST]"""
    else:
        prompt = f""" Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text}"""
    return prompt

def simple_batching(items, batch_size=5):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def read_data(fname, max_examples=None):
    examples = []
    with gzip.open(fname, "rt", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
            if max_examples and len(examples) >= max_examples:
               break
    return examples

def load_model(model_name_or_path, cache_dir, quantization):
    if quantization==4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,       
        )
    else:
        None
        #TODO
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=quantization_config, cache_dir=cache_dir)
    print("model loaded")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def sliding_window(tokens, prompt_size, window_size, input_text):
    start = randint(prompt_size, max(prompt_size, tokens["input_ids"].size()[1] - window_size))
    truncated_tokens = tokens["input_ids"][:,start:start+window_size]
    truncated_attention_mask = tokens["attention_mask"][:,start:start+window_size]
    mapping = tokens["offset_mapping"][0]
    truncated_text_start, truncated_text_end = mapping[start], mapping[start + window_size]
    truncated_text = input_text[0][truncated_text_start:truncated_text_end]
    return { "input_ids" : truncated_tokens.to(device), "attention_mask" : truncated_attention_mask.to(device)}, truncated_text

def align(input_text, output_text):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    alignments = aligner.align(input_text.replace("\n", ""), output_text.replace("\n", ""))
    alignment = alignments[0]
    return alignment

#def process_alignment(alignment):
 #   align_str = str(alignment.split("\n"))
  #  return align_str[0], align_str[1], align_str[2]

def extract_aligned_text(alignment):       #still in progress
    base_text, alignment_string, query_text = str(alignment).split('\n')
    matches = list(re.finditer(r'\|+', alignment_string))
    refined_matches = []

    #checking if text checks out but that doesn't sound necessary. at least we make sure the alignment has been processed correctly
    for match in matches:
        start, end = match.start(), match.end()
        if base_text[start:end] == query_text[start:end]:  
            refined_matches.append(match)
    if not refined_matches:
        return ""  

    last_match = max(refined_matches, key=lambda m: m.end())
    first_match = min(refined_matches, key=lambda m: m.start())
    start, end = first_match.start(), last_match.end()

    #aligned_text = base_text[start:end]
    return start, end 

def adjust_indices_to_token_boundaries(original_text, index, tokenizer):
    tokens = tokenizer.tokenize(original_text)
    cumulative_length = 0
    for token in tokens:
        token_length = len(token.replace("##", ""))  # replace special tokens from tokenizer ? 
        cumulative_length += token_length
        if cumulative_length > index:
            return original_text.index(token), token
    return index, None  # in case the index is at the end of the text

def main():
    def objective(trial):
        window_size = trial.suggest_int('window_size', 10, 1000) 
        temperature = trial.suggest_float('temperature', 0.7, 1.0) 
        do_sample = trial.suggest_categorical('do_sample', [True, False])
        beam_search = trial.suggest_categorical('beam_search', [True, False])
        num_beams = trial.suggest_int('num_beams', 2, 6) 
        examples = read_data(args.input_file, max_examples=args.max_examples)
        generated_outputs = []
        text_inputs = [prepare_text_input(e) for e in examples] 
        prompt_size = get_prompt_info(tokenizer)["prompt_token_size"]
        batch_size = args.batch_size 
        for batch in simple_batching(text_inputs, batch_size=batch_size):
            raw_encoded = tokenizer(batch, return_tensors="pt", return_offsets_mapping=True, padding=True)
            encoded = sliding_window(raw_encoded, prompt_size, window_size=window_size, batch)
            generated_ids = model.generate(**encoded, max_new_tokens=2000, pad_token_id=tokenizer.eos_token_id, temperature=temperature, do_sample=do_sample, num_beams=num_beams)
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for example in decoded:
                if args.model =="mistralai/Mixtral-8x7B-Instruct-v0.1":
                    generated = example.split("[/INST]")[-1].strip() #TODO: hacky, check if generate() can be made to return only the generated part, or take the prompt length from inputs
                else:
                    generated = example
                generated_outputs.append(generated)

        # adjusted_start_index, _ = adjust_indices_to_token_boundaries(output_text, start_index, tokenizer)
        # adjusted_end_index, _ = adjust_indices_to_token_boundaries(output_text, end_index, tokenizer)


        # alignment of hypothesis and correct texts 
        references=[]
        for i in range(len(examples)):
            alignment=align(input_text=examples[i]["output"], output_text=generated_outputs[i])
            base_text, alignment_string, query_text = str(alignment).split('\n')
            start, end = extract_aligned_text(base_text, alignment_string, query_text)
            references.append(base_text[start:end].replace("-", ""))     #that probably has to be improved

                
        metrics = calculate_metrics(predictions=generated_outputs, references=references)
        huggingface_character_wer = float(metrics["character_wer"])
        return huggingface_character_wer
    
    model, tokenizer = load_model(args.model, cache_dir, quantization=args.quantization)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.ntrials)
    print('Best parameters:', study.best_params)
    print('Best CER:', study.best_value)
    for trial in study.trials:
        print(trial.value, trial.params)

if __name__ == '__main__': 
    main()