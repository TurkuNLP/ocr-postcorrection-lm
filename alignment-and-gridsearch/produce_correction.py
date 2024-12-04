import json 
import argparse
import sys
import gzip
import datetime as dt
from transformers import AutoTokenizer

sys.path.append('/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch')
import alignment.utils as astral
import text_correction.utils as corrector

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, default="test.jsonl", help="Output file for results") 
parser.add_argument("--input_file", type=str, default="/scratch/project_2005072/cassandra/ocr-postcorrection-data/English/SEGMENTED_DOCUMENTS/en_dev_sample200.jsonl.gz", help="evaluate a custom set.")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name or path")

parser.add_argument("--quantization", type=str, default="q4_0", help="quantization of the model")
parser.add_argument("--window_size", type=int, default=0, help="size of the text segments corrected at once in tokens. set to 0 if you wish to correct the whole documents at once. Setting to 0 will by default evaluate on the test set from the SEGMENTED_DOCUMENTS subrepository when available, unless the input_file has been set manually. ")
parser.add_argument("--overlap_percentage", type=float, default=0.3, help="overlap between text segments. ignored if window_size or post processing are set to 0.")

parser.add_argument("--temperature", type=float, default=0.7, help="temperature for the model")

parser.add_argument("--mirostat", type=int, default=0, help="enable mirostat. 0 disabled. 1 enabled. 2 enabled with newer version.")
parser.add_argument("--mirostat_tau", type=float, default=5.0, help="for control of perplexity")
parser.add_argument("--top_k", type=int, default= 40, help="top k used in the model. incompatible with mirostat options.")
parser.add_argument("--top_p", type=float, default= 0.9, help="top p used in the model. presumably incompatible with mirostat options.")

parser.add_argument("--open_gap_score", type=float, default=-1, help="alignment parameter")
parser.add_argument("--extend_gap_score", type=float, default=-0.5, help="alignment parameter")
parser.add_argument("--pp_degree", type=int, default=1, help="from 0 to 3, defines the level or number of steps of post-processing")
args = parser.parse_args()

def main(args):
    with open("/scratch/project_2005072/cassandra/ocr-postcorrection-lm/trashbin/huggingface_key.txt", "r") as f:
        access_token = f.readline()
    cache_dir="/scratch/project_2005072/cassandra/.cache/"
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", cache_dir=cache_dir, token=access_token)
    
    model_options = {
        "temperature": args.temperature, 
        "mirostat": args.mirostat,
        "mirostat_tau": args.mirostat_tau,
    }

    text_options = {
        "window_size": args.window_size, 
        "overlap_percentage": args.overlap_percentage,
    }

    aligner_options = {
        "open_gap_score": args.open_gap_score, 
        "extend_gap_score": args.extend_gap_score,
    }

    input_file = args.input_file
    print("\n\nRunning correction on " + input_file + "\n")
    print("model_options :" + str(model_options) + "\n")
    print("text_options :" + str(text_options) + "\n")
    print("aligner_options :" + str(aligner_options) + "\n\n")

    with open(input_file, "r") as g:
        count=0
        for line in g:    
            count+=1

            book = json.loads(line)
            timer_start=dt.datetime.now()

            document=[book["input_1"], book["input_2"]]  #for final experiment
            
            corrected_text, corrected_text_not_aligned = corrector.correct_document(document, model=args.model, tokenizer=tokenizer,
                                                        model_options=model_options,
                                                        text_options=text_options, 
                                                        aligner_options=aligner_options, 
                                                        quantization=args.quantization,
                                                        pp_degree=args.pp_degree)
            timer_total = round((dt.datetime.now()-timer_start).total_seconds(), 1)
            
            with open( "/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch/scripts/"+args.output_file, "at" ) as f:
                
                json_content = {         "originals": { "input_1": book["input_1"], 
                                             "input_2": book["input_2"], 
                                            "output_1": book["output_1"],
                                            "output_2": book["output_2"],
                                            "model_correction_not_aligned": {"1": corrected_text_not_aligned[0], 
                                                                             "2": corrected_text_not_aligned[1] } , 
                                             "model_correction_aligned": {"1": corrected_text[0], 
                                                                             "2": corrected_text[1] } , 
                                            "model_correction_aligned_joined": ' '.join(corrected_text) }, 
                                "time_taken": timer_total
                               }                   
                print(json.dumps(json_content), file=f)

if __name__ == "__main__":
    main(args)


