import json 
import argparse
import optuna
import sys
import gzip
import datetime as dt

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
args = parser.parse_args()

def objective(trial):
    # Model Parameters
    temperature = trial.suggest_float("temperature", 0.0, 2.0)
    mirostat = trial.suggest_int("mirostat", 0, 2)
    mirostat_tau = trial.suggest_float("mirostat_tau", 0, 10)

    model_options = {
        "temperature": temperature, 
        "mirostat": mirostat,
        "mirostat_tau": mirostat_tau,
    }

    ###################################
    #FIXING OPTIONS FOR 1ST EXPERIMENT#
    ###################################
    
    # using default assessed options for text preprocessing and alignment options
   
    text_options = {
        "window_size": 300, #median is 333 tokens on TCP dev set with llama3 tokenizer
        "overlap_percentage": 0.35,
    }

    aligner_options = {
        "open_gap_score": -1, 
        "extend_gap_score": -0.5,
    }

    ##################################
    
    overall_metrics=[]
    
    with gzip.open(args.input_file, "r") as f:
        count=0
        
        for line in f:         
            book = json.loads(line)
            timer_start=dt.datetime.now()
            corrected_text = corrector.correct_document(book["input"], model=args.model, 
                                                        model_options=model_options,
                                                        text_options=text_options, 
                                                        aligner_options=aligner_options)
            timer_total = round((dt.datetime.now()-timer_start).total_seconds(), 1)
            
            metrics = calculate_metrics(predictions=[corrected_text], references=[book["output"]])
            original_metrics = calculate_metrics(predictions=[book["input"]], references=[book["output"]])
            with open( "/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch/gridsearch_output/"+args.output_file, "at" ) as f:
                normalized_text = astral.suppress_format(book["input"])
                normalized_response = astral.suppress_format(corrected_text)
                
                alignment, alignment_array = astral.align(normalized_text["text"], normalized_response["text"])
                alignment_string = astral.craft_alignment_string(alignment)
                
                json_content = { "alignments": { "aligned_ocr" : alignment[0], 
                                                 "aligned_correction" : alignment[1],
                                                 "alignment_string": alignment_string },
                                 "originals": { "input": book["input"], 
                                                "output": book["output"],
                                                "model_correction": corrected_text }, 
                                 "scores": { "model_vs_GT": metrics, 
                                             "input_vs_GT": original_metrics },  
                                 "parameters": trial.params,
                                 "time_taken": timer_total }
                
                print(json.dumps(json_content), file=f)
                
            overall_metrics.append( (original_metrics["median"]["cer"]-metrics["median"]["cer"])/original_metrics["median"]["cer"] )
            count+=1

            if count>=1:
                break

    overall_metrics.sort()

    median = overall_metrics[int(len(overall_metrics)/2)]

    return median

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.ntrials)

if __name__ == "__main__":
    main()
