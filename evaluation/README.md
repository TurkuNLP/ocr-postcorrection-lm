# eval_metrics.py usage

For evaluating the similarity of two texts or collections of text. Can be used from the command line or as a function call.

To calculate the similarity, the texts are normalized, and optionally lowercased by using casefolding. By default, CER (Character Error Rate) is used as the evaluation metric. Also WER (Word Error Rate), CharacTER, or all three together can be used.

## Command line usage

Expects the predictions and references to be stored in .jsonl files. By default, reads the text stored as a value for a key called "output". Optionally, a different key can be given as a parameter.

Uses CER for evaluation, unless a different metric is given.

The text can be casefolded (aggressively lowercased) before feeding to the evaluation metric by giving the -l | --lower flag.

**Example usage**

```
eval_metrics.py -p predictions_file.jsonl [text_key] -r references_file.jsonl [text_key] [-m {cer,wer,character,all}]
```

```
Required:
    -p | --predictions <path or filename> [text_key]
    -r | --references <path or filename> [text_key]

Optional:
    -m | --metric {cer,wer,character,all}
    -l | --lower
```

## Function call usage

### `calculate_metrics()`

Use if the predictions and references are stored as lists of strings.

Takes 4 keyword arguments:

- `predictions=` (a list that stores the texts as strings)
- `references=` (a list that stores the texts as strings)
- `metric="cer"` (_optional_, `cer`|`wer`|`character`|`all`)
- `lowercase="False"` (_optional_, `True` | `False`)

**Example usage**

```
from eval_metrics import calculate_metrics

predictions = ["This is a sentence.", "This is sentence number 2."]
references = ["This is the correct sentence.", "This is sentence number 2."]

scores = calculate_metrics(predictions=predictions, references=references)
print(scores)

### Output: {'cer': 0.2}
```

### `evaluate()`

Use if the predictions and references are stored in `.jsonl` files.

Takes 4 keyword arguments:

- `p_args=` (a string that stores the path or file name OR a list where the first element is the path/file name and the second is the key to the text (by default: `output`))
- `r_args_=` (a string that stores the path or file name OR a list where the first element is the path/file name and the second is the key to the text (by default: `output`))
- `metric="cer"` (_optional_, `cer`|`wer`|`character`|`all`)
- `lowercase="False"` (_optional_, `True` | `False`)

**Example usage**

If the text is stored under the default key (`output`):

```
from eval_metrics import evaluate

path_to_predictions = "your/path/to/predictions/here.jsonl"
path_to_references = "your/path/to/references/here.jsonl"

scores = evaluate(p_args=path_to_predictions, r_args=path_to_references)

```

If the text is stored under different key than the default:

```
from eval_metrics import evaluate

path_to_predictions = "your/path/to/predictions/here.jsonl"
key_to_predictions = "input"

path_to_references = "your/path/to/references/here.jsonl"

scores = evaluate(p_args=[path_to_predictions, key_to_predictions], r_args=path_to_references)

```