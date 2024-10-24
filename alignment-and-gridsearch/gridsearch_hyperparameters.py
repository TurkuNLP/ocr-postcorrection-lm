import json 
import argparse
import optuna
import sys
import gzip
import datetime as dt
from transformers import AutoTokenizer

sys.path.append('/scratch/project_2005072/cassandra/ocr-postcorrection-lm/evaluation')
from eval_metrics import calculate_metrics

sys.path.append('/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch')
import alignment.utils as astral
import text_correction.utils as corrector

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/scratch/project_2005072/cassandra/ocr-postcorrection-lm/trashbin/smol_sample.jsonl.gz")
parser.add_argument("--output_file", type=str, default="test.jsonl", help="Output file for results") 
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name or path")
parser.add_argument("--ntrials", type=int, default=100, help="Number of trials in the optuna gridsearch")
parser.add_argument("--options", type=int, help="0, for mirostat without eta, 1 for the mirostat with eta, 2 for the topk topp run")
args = parser.parse_args()

books=[]
with gzip.open(args.input_file, "r") as input_file:
    for line in input_file:
        books.append(json.loads(line))

length_of_all_chunks = sum([len(book["input"]) for book in books])

with open("/scratch/project_2005072/cassandra/ocr-postcorrection-lm/trashbin/huggingface_key.txt", "r") as f:
    access_token = f.readline()
cache_dir="/scratch/project_2005072/cassandra/.cache/"
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", cache_dir=cache_dir, token=access_token)

#overall_original_metrics=calculate_metrics(predictions=[book["input"] for book in books], references=[book["output"] for book in books])

def objective(trial):
    # Model Parameters
    temperature = trial.suggest_float("temperature", 0.0, 2.0)
    
    mirostat_tau = 0
    mirostat_eta = 0 #initializing in case they don't get assigned - if mirostat == 0.
    
    if args.options==0:

        mirostat = trial.suggest_int("mirostat", 0, 2)
        if mirostat!=0:
            mirostat_tau = trial.suggest_float("mirostat_tau", 0, 10)

        model_options = {
            "temperature": temperature, 
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
        }
            
    if args.options==1:

        mirostat = trial.suggest_int("mirostat", 0, 2)
        if mirostat!=0:
            mirostat_tau = trial.suggest_float("mirostat_tau", 0, 10)
            mirostat_eta = trial.suggest_float("mirostat_eta", 0.01, 0.2)

        model_options = {
            "temperature": temperature, 
            "mirostat": mirostat,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta
        }
        

    if args.options==2:

        top_k = trial.suggest_int("top_k", 10, 100)
        top_p = trial.suggest_float("top_p", 0.5, 0.95) 
    
        model_options = {
            "temperature": temperature, 
            "top_k": top_k,
            "top_p": top_p,
        }
        

    ###################################
    #FIXING OPTIONS FOR 1ST EXPERIMENT#
    ###################################
    
    # using default assessed options for text preprocessing and alignment options
   
    text_options = {
        "window_size": 300, #median is 333 tokens on TCP dev set with llama3 tokenizer
        "overlap_percentage": 0.35, # not used in the first experiment 
    }

    aligner_options = {
        "open_gap_score": -1, 
        "extend_gap_score": -0.5,
    }

    ##################################
    
    overall_metrics=[]
    
    
    count=0
    corrected_texts = []
    
    for book in books: 
        timer_start=dt.datetime.now()
        corrected_text, corrected_text_not_aligned = corrector.correct_document(book["input"], model=args.model, tokenizer=tokenizer, 
                                                    model_options=model_options,
                                                    text_options=text_options, 
                                                    aligner_options=aligner_options)
        
        timer_total = round((dt.datetime.now()-timer_start).total_seconds(), 1)
        
        metrics = calculate_metrics(predictions=[corrected_text], references=[book["output"]])
        original_metrics = calculate_metrics(predictions=[book["input"]], references=[book["output"]])
        
        with open( "/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch/gridsearch_output/"+args.output_file, "at" ) as output_file:
            normalized_text = astral.suppress_format(book["input"])
            normalized_response = astral.suppress_format(corrected_text_not_aligned[0])
            
            alignment, alignment_array = astral.align(normalized_text["text"], normalized_response["text"])
            alignment_string = astral.craft_alignment_string(alignment)
            
            json_content = { "alignments": { "aligned_ocr" : alignment[0], 
                                             "aligned_correction" : alignment[1],
                                             "alignment_string": alignment_string },
                             "originals": { "input": book["input"], 
                                            "output": book["output"],
                                            "model_correction_not_aligned": corrected_text_not_aligned[0],
                                            "model_correction":corrected_text}, 
                             "scores": { "model_vs_GT": metrics["micro"]["cer"], 
                                         "input_vs_GT": original_metrics["micro"]["cer"]},  
                             "parameters": trial.params,
                             "time_taken": timer_total }
            
            print(json.dumps(json_content), file=output_file, flush=True)
        
        count+=1
        #corrected_texts.append(corrected_text)
        single_improvement = max(min(( original_metrics["micro"]["cer"] - metrics["micro"]["cer"] ) / original_metrics["micro"]["cer"], 1), -1) #capping between -1 and 1 
        overall_metrics.append(float(single_improvement)*len(book["input"]))
        
        #if count>1:
         #   break

    #overall_lm_metrics = calculate_metrics(predictions=corrected_texts, references=[book["output"] for book in books])
    #improvement = (overall_original_metrics["median"]["cer"] - overall_lm_metrics["median"]["cer"]) / overall_original_metrics["median"]["cer"]
    
    improvement = sum(overall_metrics) / length_of_all_chunks
    
    return improvement

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.ntrials)

if __name__ == "__main__":
    main()
