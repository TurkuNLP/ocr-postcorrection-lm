import json 
import argparse
import optuna
from eval_metrics import calculate_metrics

sys.path.append('/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch')

import alignment.utils as astral
import text_correction.utils as corrector

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="/scratch/project_2000539/jenna/ocr-correction/by_page_dev_slim.jsonl.gz")
parser.add_argument("--output_file", type=str, default="out.jsonl", help="Output file for results, not implemented yet") #TODO : clean output file ?
parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name or path")
parser.add_argument("--ntrials", type=int, default=5, help="Number of trials in the optuna gridsearch")
args = parser.parse_args()

def objective(trial):
    temperature = trial.suggest_float("temperature", 0.0, 10.0)
    mirostat_tau = trial.suggest_float("mirostat_tau", 0, 7)
    window_size = trial.suggest_int("window_size", 10, 1000)
    overlap_percentage = trial.suggest_float("overlap_percentage", 0.05, 0.95)
    
    with open(args.input_file, "r") as f:
        count=0
        
        for line in f:         
            book = json.loads(line)
            corrected_text = corrector.correct_document(book["input"], model="meta-llama/Meta-Llama-3.1-8B")
            
            with open( "./corrections/" + doc_name + ".jsonl", "wt" ) as f:
                normalized_text = astral.suppress_format(book["input"])
                normalized_response = astral.suppress_format(corrected_text)
                
                alignment, alignment_array = astral.align(normalized_text["text"], normalized_response["text"])
                alignment_string = astral.craft_alignment_string(alignment)
                
                json_content = {"alignments":{"aligned_ocr" : alignment[0], 
                                              "aligned_correction" : alignment[1] ,
                                              "alignment_string": alignment_string} ,
                               "originals": { "input": book["input"], 
                                            "output":book["output"],
                                            "corrected_input": corrected_text}
                               }                   
                print(json.dumps(json_content), file=f)
            
            metrics = calculate_metrics(predictions=corrected_text, references=book["input"])
    
            overall_metrics += metrics["cer"]
            count+=1

        overall_metrics/=count

    return overall_metrics

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.ntrials)

if __name__ == "__main__":
    main()
