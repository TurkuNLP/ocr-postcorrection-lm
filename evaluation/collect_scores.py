import os
import argparse
import re
from pathlib import Path
import json
from eval_metrics import calculate_metrics
import csv

def find_matching_files(regex):
    # Compile the regular expression pattern
    pattern = re.compile(regex)
    
    # Iterate through all files in the directory and subdirectories
    file_paths = []
    for file_path in Path(os.getcwd()).rglob('*'):  # rglob('*') recursively finds all files
        if file_path.is_file() and pattern.search(str(file_path)):  # Check if file matches regex
            file_paths.append(file_path)
    
    return sorted(file_paths)

def return_matching_files(file_paths):
    while True:
        print(f"Found the following {len(file_paths)} files matching the regex:")
        for file in file_paths:
            print(file)
        command = input(f"Are you happy to proceed with these {len(file_paths)} files? (y/n/e) (y (yes): proceed / n (no): give new regex / e: exit)\n")
        if command == "y" or command == "Y":
            return file_paths
        elif command == "n" or command == "N":
            new_regex = input("Please provide a new pattern to match:\n")
            file_paths = find_matching_files(new_regex)
        elif command == "e" or command == "E":
            print("Exiting the program.")
            exit(0)
        else:
            print("Please type either y (to proceed), n (to give a new regex) or e (to exit).")

def get_keys(obj):
    keys = []
    get_keys_recursively(obj, keys)
    return keys

def get_keys_recursively(value, keys_list:list):
    
    # If the value is a dictionary, recurse through its keys
    if isinstance(value, dict):
        for key, sub_value in value.items():
            keys_list.append(key)
            get_keys_recursively(sub_value, keys_list)

def extract_model_info(filename:str):

    model_name = filename.rsplit("/", 1)[-1]

    # Delete word 'final' so it doesn't confuse the language detection part
    if "final" in model_name:
        model_name = model_name.replace("_final", "")

    # Determine the language
    if "_en" in model_name or "english" in model_name:
        lang = "en"
    elif "_fi" in model_name or "finnish" in model_name:
        lang = "fi"

    # Determine the type of the model variant
    if "q4_0" in model_name or "gpt-4o" in model_name:
        model_name = model_name.replace("q4_0", "")
        model_type = "q4_0"
    elif "fp16" in model_name:
        model_name = model_name.replace("fp16", "")
        model_type = "fp16"
    elif "fancy" in model_name and not "fancy2" in model_name and not "fancy3" in model_name and not "fancy4" in model_name:
        model_type = "fancy"
    elif "fancy2" in model_name:
        model_type = "fancy2"
    elif "fancy3" in model_name:
        model_type = "fancy3"
    elif "fancy4" in model_name:
        model_type = "fancy4"
    else:
        model_type = "basic"
    
    model_name = model_name.split("_", 1)[0]
    
    return model_name, lang, model_type

def initialize_default_keys():
    ocr = "input"
    model_correction = "model_correction_aligned"
    gt = "output"
    return [ocr, model_correction, gt]

def read_jsonl(filename):

    model_name, lang, model_type = extract_model_info(filename)

    default_keys = initialize_default_keys()

    texts = {"ocr_texts": [],
             "model_texts": [],
             "gt_texts": [],
             "model_texts_not_aligned": []
            }

    if model_type == "q4_0" or "gpt-4o" in model_name:
        default_keys.append("model_correction_not_aligned")
    
    with open(filename, "rt", encoding="utf-8") as f:
        first_line = True
        for line in f:
            example = json.loads(line)
            if first_line:
                assert isinstance(example, dict), f"The first line of the file {filename} didn't contain a jsonl object."
                first_line = False
                keys = get_keys(example) #TODO: this is really not needed, I think I have forgotten the original idea behind this
            
            if "originals" in keys:
                example = example["originals"]
            
            # If only 3 keys are present, texts["model_texts_not_aligned"] will not be included
            for key, texts_list in zip(default_keys, texts.values()):

                # Quantized models
                if model_type == "q4_0" or model_type == "fp16" or "gpt-4o" in model_name:
                    text = example[key]
                
                # Basic & all Fancy methods
                else:
                    try:
                        # used for 'input' and 'output' keys
                        text = " ".join([example[f"{key}_1"], example[f"{key}_2"]])
                    except KeyError:
                        # used for 'model_correction_aligned' key
                        try:
                            # Basic   
                            if model_type == "basic" and key == "model_correction_aligned":
                                # manually join the parts as the joined version in the file doesn't include a space between
                                text = " ".join([example[key]["1"], example[key]["2"]])
                            
                            # Fancy {2,3,4}
                            else:
                                # Do not use the above method for 'fancy' files, as this would produce an extra space in the end,
                                # as the second parts are empty (at least in fancy and fancy2?)
                                # Use 'model_correction_aligned_joined' instead
                                text = example[f"{key}_joined"]
                        except KeyError:
                            print(f"No such keys: {key}_1, {key}_2, [{key}]['1'], [{key}]['2']")
                            exit(1)

                # Extract the correct text within the above statements, and add to the specified list here in the end
                texts_list.append(text)
    
    assert all(all(isinstance(item, str) for item in value) for value in texts.values()), "Not all elements in the lists are strings"
    return model_name, model_type, lang, texts

def load_scores_object(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    else:
        # Initialize an empty object with the correct structure if no file is found
        return initialize_scores_object()

def save_scores_object(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
        
def update_scores_object(model_info, scores, filename:str):

    if not filename.endswith(".json"):
        filename += ".json"
    print(f"Updating the scores in file {filename}")

    model_name, model_type, lang, modernized, normalized = model_info
    scores_object = load_scores_object(filename)

    for metric in scores["improvement"].keys():

        scores_key = f"{lang}_{metric}"
        if lang == "fi" and modernized:
            scores_key = f"{lang}_{metric}_mod"
        
        if model_name == "gpt-4o" and not normalized:
            scores_key = f"{lang}_{metric}_no_norm"

        scores_object[model_name][model_type][scores_key] = scores["improvement"][metric]["weighted average"]

        if metric == "cer":
            if lang == "fi" and not modernized: 
                continue # only save the improvements for Finnish if modernization (w -> v) is on, so skip if not modernized
            positive = len([score for score in scores["improvement"][metric]["individual"] if score > 0])
            total = len(scores["improvement"][metric]["individual"])
            scores_object[model_name][model_type][f"num_improvements_{lang}"] = f"{positive}/{total}"

        # Assuming that the metric keys are the same for the following subdictionaries
        for subdict in ("micro", "mean", "median"):
            if lang == "fi" and not modernized:
                continue # only save the absolute results for Finnish if modernization (w -> v) is on, so skip if not modernized
            scores_object[model_name][model_type][f"{lang}_{subdict}_{metric}"] = scores[subdict][metric]

    save_scores_object(filename, scores_object)

def initialize_model_object(model):
    model = {}
    for model_type in ("q4_0", "no_alignment", "fp16", "basic", "fancy", "fancy2", "fancy3", "fancy4"):
        model[model_type] = {"en_cer": 0, "en_wer": 0, "fi_cer": 0, "fi_wer": 0, "fi_cer_mod": 0, "fi_wer_mod": 0}
    return model

def initialize_scores_object():

    models = {}

    # Order present in Overleaf
    model_order = ["Meta-Llama-3-8B-Instruct",
                   "Meta-Llama-3.1-8B-Instruct",
                   "Meta-Llama-3.1-70B-Instruct",
                   "Mixtral-8x7B-Instruct-v0.1",
                   "gemma-2-9b-it",
                   "gemma-2-27b-it",
                   "gpt-4o"]
    
    for model in model_order:
        models[model] = initialize_model_object(model)

    #sorted_models = {key: models[key] for key in model_order if key in models}
    return models

def extract_filenames(read_from):
    if os.path.exists(read_from) and os.path.isfile(read_from):
        return [read_from]
    else:
        return return_matching_files(find_matching_files(read_from))

def write_individual_scores_to_file(scores, model_name, model_type, lang, aligned=True, write_to="cer_wer_inspection"):

    if not aligned:
        filename = f"{write_to}_{model_name}_{model_type}_{lang}_not-aligned.csv"
    else:
        filename = f"{write_to}_{model_name}_{model_type}_{lang}.csv"
    
    def yield_results():
        for cer, wer, cer_impv, wer_impv, orig_cer, orig_wer in zip(scores["individual"]["cer"],
                                                                    scores["individual"]["wer"],
                                                                    scores["improvement"]["cer"]["individual"],
                                                                    scores["improvement"]["wer"]["individual"],
                                                                    scores["original"]["cer"],
                                                                    scores["original"]["wer"]):
            #print(orig_cer, cer, cer_impv)
            yield {"orig_cer":f"{orig_cer*100:.1f}",
                   "cer": f"{cer*100:.1f}",
                   "cer_improvement": f"{cer_impv*100:.1f}",
                   "orig_wer": f"{orig_wer*100:.1f}",
                   "wer": f"{wer*100:.1f}",
                   "wer_improvement": f"{wer_impv*100:.1f}",
                   }

    with open(filename, "w") as file:
        fieldnames = ["orig_cer", "cer", "cer_improvement", "orig_wer", "wer", "wer_improvement"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(yield_results())

def run_evaluation_and_inspect(file_path, normalized, modernized, remove_newlines=False):
    
    model_name, model_type, lang, texts = read_jsonl(str(file_path))
    ocr_texts, model_texts, gt_texts, model_texts_not_aligned = texts.values()

    def get_scores(predictions):
        return calculate_metrics(predictions=predictions,
                                 references=gt_texts,
                                 originals=ocr_texts,
                                 metric="all",
                                 modernize=modernized,
                                 do_normalize=normalized,
                                 remove_newlines=remove_newlines)
    
    scores = get_scores(model_texts)
    write_individual_scores_to_file(scores=scores, model_name=model_name, model_type=model_type, lang=lang)

    if model_texts_not_aligned:
        scores = get_scores(model_texts_not_aligned)
        write_individual_scores_to_file(scores=scores, model_name=model_name, model_type=model_type, lang=lang)


def run_evaluation_and_update(scores_object, model_name, model_type, lang, predictions, references, originals, normalized, modernized, remove_newlines=False):
    
    scores = calculate_metrics(predictions=predictions,
                               references=references,
                               originals=originals,
                               metric="all",
                               modernize=modernized,
                               do_normalize=normalized,
                               remove_newlines=remove_newlines)
    
    update_scores_object(((model_name, model_type, lang, modernized, normalized)), scores, scores_object)

def evaluate_and_update_all(scores_object, file_path, remove_newlines):

    model_name, model_type, lang, texts = read_jsonl(str(file_path))
    ocr_texts, model_texts, gt_texts, model_texts_not_aligned = texts.values()

    # Default values
    normalized = True
    modernized = False if lang == "en" else True

    # Base case
    run_evaluation_and_update(scores_object=scores_object, model_name=model_name, model_type=model_type, lang=lang,
                              predictions=model_texts, references=gt_texts, originals=ocr_texts,
                              normalized=normalized, modernized=modernized, remove_newlines=remove_newlines)

    # If also the texts without post-processing ('model_texts_not_aligned') are returned, run evaluation on them
    if model_texts_not_aligned:
        run_evaluation_and_update(scores_object=scores_object, model_name=model_name, model_type="no_alignment", lang=lang,
                                  predictions=model_texts_not_aligned, references=gt_texts, originals=ocr_texts,
                                  normalized=normalized, modernized=modernized, remove_newlines=remove_newlines)

    # For Finnish: also run without modernization
    if lang == "fi":
        run_evaluation_and_update(scores_object=scores_object, model_name=model_name, model_type=model_type, lang=lang,
                                  predictions=model_texts, references=gt_texts, originals=ocr_texts,
                                  normalized=normalized, modernized=False, remove_newlines=remove_newlines)
    
    # only run the following for English texts
    else:

        # For GPT-4o, also run scores for texts without normalization (for discussion part)
        # TODO: change if we do want to run these settings for Finnish, too
        if model_name == "gpt-4o":
            run_evaluation_and_update(scores_object=scores_object, model_name=model_name, model_type=model_type, lang=lang,
                                      predictions=model_texts, references=gt_texts, originals=ocr_texts,
                                      normalized=False, modernized=modernized, remove_newlines=remove_newlines)


def main(args):

    SCORES_OBJECT = args.scores_file

    file_paths = extract_filenames(args.read_from)
    total_files = len(file_paths)

    for i, file_path in enumerate(file_paths, 1):
        
        print(f"Now reading file {file_path}")

        if args.inspect:
            run_evaluation_and_inspect(file_path=file_path, remove_newlines=args.no_newlines)
            continue
        
        evaluate_and_update_all(scores_object=SCORES_OBJECT, file_path=file_path, remove_newlines=args.no_newlines)
        
        print(f"{i}/{total_files} files processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("read_from",
                        type=str,
                        help="Filename or a regular expression for matching multiple files.")
    parser.add_argument("--scores_file",
                        "-s",
                        type=str,
                        default="scores_object.json",
                        help="Filename for the scores object.")
    parser.add_argument("--inspect",
                        "-i",
                        action="store_true",
                        help="Write individual CER and WER results into a file for inspection.")
    parser.add_argument("--no_newlines",
                        "-n",
                        action="store_true",
                        default=False,
                        help="Collect scores with newlines removed from all texts.")
    args = parser.parse_args()

    main(args)