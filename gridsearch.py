
import json
import argparse
import transformers
import torch
import bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import gzip
import unicodedata
from eval_metrics import calculate_metrics
import optuna

print(transformers.__version__)
print(bitsandbytes.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/scratch/project_2000539/jenna/ocr-correction/by_page_dev_slim.jsonl.gz")
parser.add_argument("--output_file", type=str, required=True, help="Output file for the generated text")
parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name or path")
parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process")
parser.add_argument("--ntrials", type=int, default=5, help="Number of trials in the optune gridsearch")
args = parser.parse_args()

device = "cuda" # the device to load the model onto
cache_dir = "/scratch/project_2005072/cassandra/.cache" # path to the cache dir where the models are

def prepare_text_input(example):
    text = example["input"]
    prompt = f"""[INST] Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n{text} [/INST]"""
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
    # TODO: quatization options
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,       
    )
    if quantization:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=quantization_config, cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def objective(trial):
    window_size = trial.suggest_int('window_size', 1, 10)  # play on window size ?
    #quantization = trial.suggest_int('quantization', 4, 8)
    quantization=True
    model, tokenizer = load_model(args.model, cache_dir, quantization=quantization)

    examples = read_data(args.input_file, max_examples=args.max_examples)

    generated_outputs = []

    batch_size = 5 # set your batch size
    for batch in simple_batching(text_inputs, batch_size=batch_size):
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        model_inputs = encoded.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=2000, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for example in decoded:
            generated = example.split("[/INST]")[-1].strip() #TODO: hacky, check if generate() can be made to return only the generated part, or take the prompt length from inputs
            generated_outputs.append(generated)

     
    metrics = calculate_metrics(predictions=[item for item in generated_outputs], references=[item["output"] for item in examples])

    return metrics

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.ntrials)
    print('Best parameters:', study.best_params)
    print('Best CER:', study.best_value)

    for trial in study.trials:
        print(trial.value, trial.params)

if __name__ == '__main__': 
    main()