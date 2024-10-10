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
    temperature = trial.suggest_float("temperature", 0.0, 10.0)
    mirostat_tau = trial.suggest_float("mirostat_tau", 0, 7)
    window_size = trial.suggest_int("window_size", 10, 5000)
    overlap_percentage = trial.suggest_float("overlap_percentage", 0.05, 0.95)

    # other variables to look through ? 

    #alignment parameters

    open_gap_score = trial.suggest_float("open_gap_score", -2.0, 0.0)
    extend_gap_score = trial.suggest_float("extend_gap_score", -2.0, 0.0)
    
    model_options = {
        "temperature": temperature, 
        "mirostat": 2,
        "mirostat_tau": mirostat_tau,
    }

    text_options = {
        "window_size": window_size, 
        "overlap_percentage": overlap_percentage,
    }

    aligner_options = {
        "open_gap_score": open_gap_score, 
        "extend_gap_score": extend_gap_score,
    }

    overall_metrics=0
    
    with gzip.open(args.input_file, "r") as f:
        count=0
        
        for line in f:         
            book = json.loads(line)
            #add time !!!
            #tokens ?
            timer_start=dt.datetime.now()
            corrected_text = corrector.correct_document(book["input"], model=args.model, 
                                                        model_options=model_options,
                                                        text_options=text_options, 
                                                        aligner_options=aligner_options)
            timer_total = round((dt.datetime.now()-timer_start).total_seconds(), 1)
            
            metrics = calculate_metrics(predictions=[corrected_text], references=[book["input"]])
            
            with open( "./gridsearch_output/"+args.output_file, "wt" ) as f:
                normalized_text = astral.suppress_format(book["input"])
                normalized_response = astral.suppress_format(corrected_text)
                
                alignment, alignment_array = astral.align(normalized_text["text"], normalized_response["text"])
                alignment_string = astral.craft_alignment_string(alignment)
                
                json_content = {"alignments":{"aligned_ocr" : alignment[0], 
                                              "aligned_correction" : alignment[1] ,
                                              "alignment_string": alignment_string},
                               "originals": { "input": book["input"], 
                                            "output":book["output"],
                                            "model_correction": corrected_text}, 
                                "scores": metrics, 
                                "parameters": trial.params,
                                "time_taken": timer_total
                               }                   
                print(json.dumps(json_content), file=f)
            
            
    
            overall_metrics += metrics["cer"]
            count+=1

        overall_metrics/=count

    return overall_metrics

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.ntrials)

if __name__ == "__main__":
    main()
