print("importing...")

from Bio.Align import PairwiseAligner
import json
import argparse
import torch
import transformers
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from random import randint
import gzip
import unicodedata
from eval_metrics import calculate_metrics
import optuna
import datetime as dt 
import re
import traceback
import logging
import alignment.utils as astral 

logging.getLogger('transformers').setLevel(logging.ERROR) ## in order to disable """A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.""", which in our case does not hold.

print("transformers version: ", transformers.__version__)
print("bitsandybtes version: ", bitsandbytes.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/scratch/project_2000539/jenna/ocr-correction/by_page_dev_slim.jsonl.gz")
parser.add_argument("--output_file", type=str, default="out.jsonl", help="Output file for results, not implemented yet") #TODO : clean output file ?
parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name or path")
parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process")
parser.add_argument("--ntrials", type=int, default=5, help="Number of trials in the optune gridsearch")
parser.add_argument("--quantization", type=int, default=4, help="Number of bits for quantization (4, 8, 16)")
parser.add_argument("--batch_size", type=int, default=1, help="Set up a batch size for batched generation") # this code does not really support batching right now
args = parser.parse_args()

device = "cuda" # the device to load the model onto
cache_dir = '/scratch/project_2005072/ocr_correction/.cache'  # path to the cache dir where the models are
access_token = "hf_x"

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) #print stuff in red 
  
def prepare_text_input(example, tokenizer, slicing=False, window_size=100):
    if not slicing:
        text = example["input"]
        prompt = f""" Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text}"""
        
        return [prompt], [text], [example["reference_number"]]
        
    if slicing:        
        prompts=[]
        texts=[]
        reference_numbers=[]
        input = example["input"]
        tokenized_text = tokenizer(input,return_offsets_mapping=True, return_tensors="pt").to(device)

        n = tokenized_text["input_ids"].size()[1]
        mapping = tokenized_text["offset_mapping"]
        number_of_slices = max(1, n//window_size)
        
        if number_of_slices > 1 :
            number_of_slices-=1
        
        for i in range(number_of_slices):
            print('i', i)
            
            try:
                text = input[int(mapping[0][i*window_size][0]):int(mapping[0][(i+1)*window_size][0])]
            except:
                text = input[int(mapping[0][i*window_size][0]):int(mapping[0][-1][1])]
                
            prompt = f""" Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text}"""
            texts.append(text)
            prompts.append(prompt)
            reference_numbers.append(example["reference_number"])

        return prompts, texts, reference_numbers
            

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
        count=0
        for line in f:
            example = json.loads(line)
            example["reference_number"]=count
            examples.append(example)
            count+=1
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

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=quantization_config, cache_dir=cache_dir, token=access_token)
        
    elif quantization==8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.float16,       
        )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=quantization_config, cache_dir=cache_dir, token=access_token)
        
    else:
        print("Warning: Quantization number not found.Please use --quantization 4 or --quantization 8 if you want to use a quantized model. Loading full model")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", cache_dir=cache_dir, token=access_token)
    print("model loaded")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer loaded")

    return model, tokenizer

def sliding_window(tokens, tokenizer, prompt_size, window_size): #cut a part of the input to see how the length of the input influences the ability of the model to correct. misleading function name

    input_text = tokenizer.decode(tokens["input_ids"][0])
    mapping = tokens["offset_mapping"][0]

    start = randint(prompt_size, max(prompt_size, tokens["input_ids"].size()[1] - window_size))
    end = min(start+window_size, tokens["input_ids"].size()[1]-2)

    test = tokenizer("<|eot_id|>", return_tensors="pt")
    size_eot = test["input_ids"].size()[0]
    #print(start, end, tokens["input_ids"].size()[1])
    truncated_tokens = torch.cat((tokens["input_ids"][:,:prompt_size-1],tokens["input_ids"][:,start:end]), dim=1)
    truncated_tokens = torch.cat((truncated_tokens,tokens["input_ids"][:,-size_eot:]), dim=1)

    truncated_attention_mask = torch.cat((tokens["attention_mask"][:,:prompt_size-1],tokens["attention_mask"][:,start:end]), dim=1)
    truncated_attention_mask = torch.cat((truncated_attention_mask,tokens["attention_mask"][:,-size_eot:]), dim=1)

    
    #print("size", mapping.size())
    #print(mapping[start], mapping[end])
    truncated_text_start, truncated_text_end = int(mapping[start][0]), int(mapping[end][0])
    #print("n:", n, truncated_text_start-n, truncated_text_end-n)
    truncated_text = input_text[truncated_text_start:truncated_text_end]
    
    return { "input_ids" : truncated_tokens.to(device), "attention_mask" : truncated_attention_mask.to(device)}, truncated_text
    




def normalize(text):
    text = " ".join(text.replace("\n", " ").split())
    return unicodedata.normalize("NFKC", text)

def get_prompt_size(tokenizer):  # for now this is only used to get to know how long the prompt is in terms of token length to not cut it when slicing the text
    messages = [
        {"role": "system", "content": "You are a useful assistant"},
        {"role": "user", "content": f""" Correct the OCR errors in the text below. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n"""},
    ]
    encoded = tokenizer.apply_chat_template(messages, return_dict=True, return_tensors="pt").to(device)
    prompt_token_size = encoded["input_ids"][0].size()[0]-1  # -1 to exclude the eot token
                                                                                 
    return prompt_token_size 

class Optimizer:
    def __init__(self):
        self.model, self.tokenizer = load_model(args.model, cache_dir, quantization=args.quantization)
        self.prompt_size = get_prompt_size(self.tokenizer)

    def objective(self, trial):
        window_size = trial.suggest_int('window_size', 10, 600) 
        do_sample = trial.suggest_categorical('do_sample', [True, False])
        temperature = None
        beam_search = False      #reset to not get warnings (doesn't actually prevent warnings, idk what to do about them)
        num_beams = None
        if do_sample:
            temperature = trial.suggest_float('temperature', 0.0, 2.0)
            beam_search = trial.suggest_categorical('beam_search', [True, False])
            if beam_search:
                num_beams = trial.suggest_int('num_beams', 2, 6) 
                
        examples = read_data(args.input_file, max_examples=args.max_examples)
        generated_outputs = []
        text_inputs = []
        base_texts_unfolded = []
        reference_numbers_unfolded = []
        for i in range(len(examples)): 
            text_input, base_text, reference_number = prepare_text_input(examples[i], self.tokenizer, slicing=True, window_size=window_size)
            text_inputs.append(text_input)
            base_texts_unfolded.append(base_text)
            reference_numbers_unfolded.append(reference_number)
            
        messages_input = []
        base_texts = []
        for list_of_elt in text_inputs:
            for elt in list_of_elt:
                messages_input.append(elt)

        for list_of_elt in base_texts_unfolded:
            for elt in list_of_elt:
                base_texts.append(elt)
                
        reference_numbers=[]
        for list_of_elt in reference_numbers_unfolded:
            for elt in list_of_elt:
                reference_numbers.append(elt)

        batch_size = args.batch_size 
        timers=[]

        truncated_inputs = []
        for batch in simple_batching(messages_input, batch_size=batch_size):

            timer_start=dt.datetime.now()
            messages = [
                {"role": "system", "content": "You are a useful assistant"},
                {"role": "user", "content": batch[0]  #bcs batch_size is only 1 for now
            },
            ]

            encoded = self.tokenizer.apply_chat_template(messages, return_dict=True,  return_tensors="pt", max_length = int(1.5*(self.prompt_size+window_size))).to(device) # return_offsets_mapping=True,
            
            #print(encoded)
            #encoded ={ "input_ids" : encoded["input_ids"].to(device), "attention_mask" : encoded["attention_mask"].to(device)}
            #encoded, truncated_text = sliding_window(raw_encoded, self.tokenizer, self.prompt_size, window_size)     
            #truncated_inputs.append(truncated_text)
            try:
                temperature #trying out if the temperature var exists
                if beam_search:
                    generated_ids = self.model.generate(**encoded, max_new_tokens=1.5*window_size, pad_token_id=self.tokenizer.eos_token_id, temperature=temperature, do_sample=do_sample, num_beams=num_beams, early_stopping=True)
                else:
                    generated_ids = self.model.generate(**encoded, max_new_tokens=1.5*window_size, pad_token_id=self.tokenizer.eos_token_id, temperature=temperature, do_sample=do_sample)
            except:
                generated_ids = self.model.generate(**encoded, max_new_tokens=1.5*window_size, pad_token_id=self.tokenizer.eos_token_id, do_sample=do_sample)
                    
                    
            decoded = self.tokenizer.batch_decode(generated_ids)
            for example in decoded:
                prRed(example)
                #if "mistralai/Mixtral" in str(args.model):
                generated = example.split("<|end_header_id|>")[-1].strip("<|eot_id|>") 
                
                generated_outputs.append(generated)

        # alignment of hypothesis and correct texts 
            timer_total = round((dt.datetime.now()-timer_start).total_seconds(), 1)
            nb_tokens = generated_ids.size()[1]
            print(f"It took {timer_total} seconds to generate {nb_tokens} tokens.")
            timers.append(timer_total)

        aligned_generated_outputs=[]
        for i in range(len(messages_input)):
            base_text = normalize(generated_outputs[i])
            try:
                n = len(normalize(examples[reference_numbers[i]]["input"]))
                alignment = astral.align(input_text=base_text, output_text=normalize(examples[reference_numbers[i]]["input"])) ## 'input' more realistic than 'output'
            except:
                alignment = ""
            try:
                start, end = astral.smoothing_window(base_text, alignment)
                aligned_generated_outputs.append(base_text[start:end]) 
            except:
                aligned_generated_outputs.append("error with alignment :" + str(alignment))

        
        references=[]
        for i in range(len(messages_input)):
            base_text = normalize(examples[reference_numbers[i]]["output"])
            if "error with alignment :" in aligned_generated_outputs[i]:
                references.append("error with alignment")
            else:
                try:
                    alignment = astral.align(input_text=base_text, output_text=aligned_generated_outputs[i]) 
                except:
                    alignment = ""
                try:
                    start, end = astral.smoothing_window(base_text, alignment)
                    references.append(base_text[start:end])  
                except:
                    references.append("error with alignment :" + str(alignment))

        try:
            clean_ref, clean_output = [], []
            penalty_count = 0
            for i in range(len(references)):
                if references[i]=="" or aligned_generated_outputs[i]=="":
                    penalty_count += 1
                    continue
                else:
                    clean_ref.append(references[i])
                    clean_output.append(aligned_generated_outputs[i])
                    
            metrics = calculate_metrics(predictions=clean_output, references=clean_ref)
            huggingface_character_wer = float(metrics["character_wer"])

            huggingface_character_wer = (huggingface_character_wer*len(clean_ref)+penalty_count)/len(references)
            
        except:
            metrics = "undefined"
            huggingface_character_wer = 10

        filename = "_".join(str(args.output_file).split("/"))

        with open(filename, "at", encoding="utf-8") as f:
            for i in range(len(messages_input)):
                try:
                    al = astral.align(input_text=base_texts[i], output_text=generated_outputs[i])
                except:
                    al = "  "
                    
                d = {"originals": 
                         {"whole_input": examples[reference_numbers[i]]["input"], 
                          "reference": examples[reference_numbers[i]]["output"]},
                     "truncated_input": base_texts[i],    
                     "output": generated_outputs[i], 
                     "aligned_output": aligned_generated_outputs[i], 
                     "aligned_reference": references[i], 
                     "total characters generated": len(generated_outputs[i]), 
                     "total tokens generated": nb_tokens, 
                     "time taken": timers[i], 
                     "scores": 
                         {"metrics": metrics, 
                          "hf_wer": huggingface_character_wer, 
                          "percentage_found": len(aligned_generated_outputs[i])/len(base_texts[i])}, 
                    "alignment": 
                         {"base":al[0], 
                          "al_str": astral.craft_alignment_string(al), 
                          "query": al[1]}, 
                     "parameters": trial.params}
                
                print(json.dumps(d, ensure_ascii=False), file=f)
        prRed("infos dumped")
        
        return huggingface_character_wer

def main():
    optimizer = Optimizer()
    study = optuna.create_study(direction='minimize')
    study.optimize(optimizer.objective, n_trials=args.ntrials)
    print('Best parameters:', study.best_params)
    print('Best CER:', study.best_value)
    for trial in study.trials:
        print(trial.value, trial.params)

if __name__ == '__main__': 
    main()

                