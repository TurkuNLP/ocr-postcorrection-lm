import openai
import sys
import json
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--apikey", type=str, help="OpenAI API key for query")
parser.add_argument("--model", type=str, help="name of the queried model")
args = parser.parse_args()

model = args.model
openai.api_key = args.apikey

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": "Bearer " + openai.api_key,
    "Content-Type": "application/json"
}

prompt = 'Correct the OCR errors in the text below. Also correct the "-" characters, which denote an unrecognized letter. Stay as close as possible to the original text. Do not rephrase. Only correct the errors. You will be rewarded. \n\n'


for line in sys.stdin:
    page = json.loads(line)
    data = {
        "model": model,
        "messages": [
            {
                "role":"system", 
                "content":"You are a helpful assistant."
            },
            {
                "role":"user",
                "content": prompt + page["input"], 
            }
        ],
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    print(json.dumps(response))


