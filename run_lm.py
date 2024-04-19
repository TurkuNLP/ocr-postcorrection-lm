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


print(transformers.__version__)
print(bitsandbytes.__version__)

# my versions are:
#4.38.1 (transformers)
#0.42.0 (bitsandbytes)

device = "cuda" # the device to load the model onto

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

def load_model(model_name_or_path, cache_dir):
    # TODO: quatization options
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,       
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", quantization_config=quantization_config, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main(args):

    # read the data
    examples = read_data(args.input_file, max_examples=args.max_examples)

    # calculate the original wer and cer (original against reference)
    original_metrics = calculate_metrics(predictions=[item["input"] for item in examples], references=[item["output"] for item in examples])
    print("Original scores:")
    print(original_metrics)

    model, tokenizer = load_model(args.model, cache_dir="/scratch/project_2000539/jenna/hf-cache")
    
    generated_outputs = []
    scores = []

    # create the text inputs (prompt, text, special tokens), apply_chat_template() is not used as it did not seem to support batching
    text_inputs = [prepare_text_input(e) for e in examples] 

    batch_size = 5  # Set your batch size
    for batch in simple_batching(text_inputs, batch_size=batch_size):
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        model_inputs = encoded.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=2000, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for example in decoded:
            generated = example.split("[/INST]")[-1].strip() #TODO: hacky, check if generate() can be made to return only the generated part, or take the prompt length from inputs
            generated_outputs.append(generated)

    print("Predicted scores:")
    metrics = calculate_metrics(predictions=generated_outputs, references=[e["output"] for e in examples])
    print(metrics)

    # print generations to file (jsonl)
    with open(args.output_file, "wt", encoding="utf-8") as f:
        for i in range(len(examples)):
            orig_metrics = calculate_metrics(predictions=[examples[i]["input"]], references=[examples[i]["output"]])
            pred_metrics = calculate_metrics(predictions=[generated_outputs[i]], references=[examples[i]["output"]])

            d = {"input": examples[i]["input"], "output": generated_outputs[i], "orig_scores": orig_metrics, "gener_scores": pred_metrics}
            
            print(json.dumps(d, ensure_ascii=False), file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="/scratch/project_2000539/jenna/ocr-correction/by_page_dev_slim.jsonl.gz")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the generated text")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name or path")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process")
    args = parser.parse_args()

    main(args)