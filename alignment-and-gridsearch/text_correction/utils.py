import json 
import sys
import traceback
import argparse
sys.path.append('/scratch/project_2005072/cassandra/ocr-postcorrection-lm/alignment-and-gridsearch')
import alignment.utils as astral

def convert_model_name_for_ollama(model, quantization="q4_0"):
    ollama_models = {
        "meta-llama/Meta-Llama-3-8B-Instruct" : "llama3:8b-instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct" : "llama3.1:8b-instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct" : "llama3.1:70b-instruct",
        "google/gemma-7b-it" : "gemma:7b-instruct",
        "google/gemma-2-9b-it" : "gemma2:9b-instruct", 
        "google/gemma-2-27b-it" : "gemma2:27b-instruct", 
        "mistralai/Mixtral-8x7B-Instruct-v0.1" : "mixtral:8x7b-instruct-v0.1"
    }

    
    ollama_model = ollama_models[model]
    
    ollama_model+= "-" + quantization
    
    return ollama_model
    
def prepare_text_input(input_text, tokenizer, window_size, overlap_percentage): 
    overlap = window_size * overlap_percentage
    device="cuda"
    texts=[]
    tokenized_text = tokenizer(input_text,return_offsets_mapping=True, return_tensors="pt").to(device)
    
    n = tokenized_text["input_ids"].size()[1]
    mapping = tokenized_text["offset_mapping"]

    end_reached = False
    i=0
    
    while not end_reached:
        start_index = int(i*(window_size-overlap))
        end_index = int((i+1)*window_size - i*overlap)
        try:
            sliced_text = input_text[int(mapping[0][start_index][0]):int(mapping[0][end_index][0])]
            i+=1
        except:
            sliced_text = input_text[int(mapping[0][start_index][0]):int(mapping[0][-1][1])]
            end_reached=True
            
        texts.append(sliced_text)

    return texts

def correct_first_time(sliced_ocr, model, model_options, aligner_options, pp_degree, quantization): ### initializing arrays of corrected texts
    corrected_texts = ["" for i in range(len(sliced_ocr))]
    responses = ["" for i in range(len(sliced_ocr))]
    ollama_model=convert_model_name_for_ollama(model, quantization)
    
    response = astral.run_ollama(ollama_model, [first_time_prompt + sliced_ocr[0]], model_options)[0]
    responses[0] = response
    
    if pp_degree>0:
        normalized_text = astral.suppress_format(sliced_ocr[0])
        normalized_response = astral.suppress_format(response)

        try:
            alignment, alignment_array = astral.align(normalized_text["text"], normalized_response["text"], aligner_options)
         
            start, end = astral.new_extract_aligned_text(normalized_response, alignment, alignment_array)
        except:
            traceback.print_exc()
            print("alignment not found - setting clip start and end to 0 and -1")
            start, end = 0, -1

        corrected_texts[0] = response[start:end]

    return corrected_texts, responses

def correct_with_feedback(previous_corrected_text, previous_ocr, ocr_continuation_text, model, model_options, aligner_options, pp_degree, quantization):
    ollama_model=convert_model_name_for_ollama(model, quantization)

    response = astral.run_ollama(ollama_model, [feedback_prompt + previous_corrected_text + """\n\n"Text to correct:" """ + ocr_continuation_text], model_options)
    response = response[0]


    print(feedback_prompt + previous_corrected_text + """\n\n"Text to correct:" """ + ocr_continuation_text, flush = True)
    start, end = 0, -1

    
    if pp_degree>0:
        if pp_degree>3:
            normalized_text = astral.suppress_format(ocr_continuation_text)
        else:
            normalized_text = astral.suppress_format(previous_ocr + ocr_continuation_text)
        normalized_response = astral.suppress_format(response)
        
        alignment, alignment_array = astral.align(normalized_text["text"], normalized_response["text"], aligner_options)
      
        start, end = astral.new_extract_aligned_text(normalized_response, alignment, alignment_array)

    return response[start:end], response

def merge_strings(A, B, aligner_options):  # merge overlapping corrected strings
    normalized_A = astral.suppress_format(A)
    normalized_B = astral.suppress_format(B)
    alignment, alignment_array = astral.align(normalized_A["text"], normalized_B["text"], aligner_options)

    a_start, a_end = astral.new_extract_aligned_text(normalized_A, alignment, alignment_array, reversed=True)

    alignment, alignment_array = astral.align(normalized_B["text"], normalized_A["text"], aligner_options)
    b_start, b_end = astral.new_extract_aligned_text(normalized_B, alignment, alignment_array, reversed=True)

    print(B)
    print(B[b_end:])
    print(b_end)
    return A[:a_end]+ B[b_end:]

def merge_overlapping_texts(corrected_texts, aligner_options):  # merge the whole corrected text back string by string 
    result = corrected_texts[0]
    for i in range(1, len(corrected_texts)):
        n = len(result)
        try:
            result = result[:-len(corrected_texts[i])]+ merge_strings(result[-len(corrected_texts[i]):], corrected_texts[i], aligner_options)
        except:
            traceback.print_exc()
            result += corrected_texts[i]
    return result

def prep(slice, cut_off=100):
    while slice[-cut_off]!=" " and slice[-cut_off]!="\n":
        cut_off+=1
    result = slice[-cut_off:]
    return result

def chop_end(correction, nb_words=5):
    count_words=0
    id = -1
    last_char = "*"
    
    while count_words<nb_words:
        if (last_char==" " or last_char=="\n") and correction[id]!=" " and correction[id]!="\n":
            count_words+=1
        last_char=correction[id]
        id-=1

    return correction[:id]
        
    
def correct_document(document, model, tokenizer, model_options, text_options, quantization="q4_0", aligner_options=None, pp_degree=1):
    ######## Assembled multi-task document correction function
    ######## Post processing degrees :
    ######## post processing set to 0 will correct the full document and NOT do ANYTHING to the model answer (no alignment, raw answer)
    ######## post processing set to 1 will correct the full document and align the model anwser to the input
    ######## post processing set to 2 or higher will assume a list of slices is being given and use correction with feedback and text alignment to generate the answer.
    ######## post processing set to 3 or higher will do same as 2 but add an overlap to the feedback.


    ####NOTE that I modified the thing so it supports pp3 right now and I would need to clean the code so it supports all the degrees properly at the same time

    #if pp_degree>1:
    #sliced_ocr = prepare_text_input(document, tokenizer, window_size=text_options["window_size"], overlap_percentage=text_options["overlap_percentage"])
    sliced_ocr=document #in the idea that i would have document=[bloc1,bloc2]
    #else:
     #   sliced_ocr=[document]
    if pp_degree==3: #very much not flexible code made for the final experiment
        sliced_ocr[1] = prep(sliced_ocr[0]) + sliced_ocr[1]
    corrected_texts, responses = correct_first_time(sliced_ocr, model, model_options, aligner_options, pp_degree, quantization)

    for i in range(1, len(sliced_ocr)):
        if pp_degree>3:
            corrected_texts[i], responses[i] = correct_with_feedback(corrected_texts[i-1], sliced_ocr[i-1], sliced_ocr[i], model, model_options, aligner_options, pp_degree, quantization)
        elif pp_degree>2:
            corrected_texts[i], responses[i] = correct_with_feedback(chop_end(corrected_texts[i-1]), sliced_ocr[i-1], sliced_ocr[i], model, model_options, aligner_options, pp_degree, quantization)
        elif pp_degree>1:
            corrected_texts[i], responses[i] = correct_with_feedback(corrected_texts[i-1], sliced_ocr[i-1], sliced_ocr[i], model, model_options, aligner_options, pp_degree, quantization)
        else:
            corrected_texts[i], responses[i] = correct_first_time([sliced_ocr[i]], model, model_options, aligner_options, pp_degree, quantization)
            corrected_texts[i], responses[i] = corrected_texts[i][0], responses[i][0]
            

    if pp_degree==2 or pp_degree==3:
        result = [merge_overlapping_texts(corrected_texts, aligner_options), ""]
        #result = [corrected_texts[-1], ""]

    
    else:
        result = corrected_texts
        
    return result, responses

first_time_prompt = """Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded.\n\n"""

#In case of only asking to correct the second text

feedback_prompt = """Correct the OCR errors in the text below, marked as "Text to correct: ". Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. The first provided text, marked as "Previous context:", is the corrected version of an OCR text. The text "Text to correct: " comes right after the text "Previous context:" in the original source. Only correct the errors and provide the corrected "Text to correct: " text.\n\n"Previous context:" """


#In case of overlap :
#feedback_prompt = """The following two texts, marked as "A:" and "B:", overlap. The second text is a continuation of the first one, but it may contain OCR errors. Correct these errors. Stay as close as possible to the original text. Reconstruct the text by merging the two overlapping texts. The overlapping parts should not be repeated. Only provide the final merged and corrected text.\n\nA:"""


#In case of no overlap :
#feedback_prompt =  """The following two texts, marked as "A:" and "B:" are linked. The second text is a continuation of the first one, but it may contain OCR errors. Correct these errors. Stay as close as possible to the original text. Reconstruct the text by merging the two texts. Only provide the final merged and corrected text.\n\nA:"""
