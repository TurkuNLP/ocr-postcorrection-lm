from collections import Counter
import json
import tqdm

#operations_counter = Counter()

#book_counter = 0
#with open("../ecco-paper-data/aligned_ecco_books_compact.jsonl", "rt", encoding="utf-8") as f:
#    for line in tqdm.tqdm(f, desc="Reading alignments"):
#        book = json.loads(line)
#        alignments = book["alignments"]
#        for c in alignments:
#            operations_counter[c] += 1
#        book_counter += 1
#        if book_counter > 1000:
#            break



operations_counter = Counter()
change_counter = Counter()
length_counter = Counter()
total_changes = 0
book_counter = 0
with open("../ecco-paper-data/aligned_ecco_books_compact.jsonl", "rt", encoding="utf-8") as f:
    for line in tqdm.tqdm(f, desc="Reading alignments"):
        book = json.loads(line)
        alignments = book["alignments"]
        orig_text = book["original"]
        corr_text = book["corrected"]
        orig = ""
        corr = ""
        for o,c,a in zip(orig_text, corr_text, alignments):
            operations_counter[a] += 1
            o = o.replace(" ", "∅")
            c = c.replace(" ", "∅")
            if a == "|":
                if orig:
                    total_changes += 1
                    length_counter[len(orig)] += 1
                    if len(orig) < 5: # no need to store change this long, will be infrequent anyway
                        change_counter.update([f"{orig} -> {corr}"])
                orig = ""
                corr = ""
            else:
                orig += o
                corr += c
        book_counter += 1
        #if book_counter > 1000:
        #    break

print("Operations:")
total = sum(operations_counter.values())
for operation, count in operations_counter.most_common():
    print(operation, f"{count:,} / {total:,} ({count/total*100:.2f}%)")
print()

print("Change lengths:")
total = sum(length_counter.values())
print(f"Total changes: {total:,}")
for length, count in length_counter.most_common(20):
    print(length, f"{count:,} / {total:,} ({count/total*100:.2f}%)")
print()
# more than 5 or 10 characters
total = sum(length_counter.values())
more_than_5 = 0
more_than_10 = 0
for length, count in length_counter.most_common():
    if length > 5:
        more_than_5 += count
    if length > 10:
        more_than_10 += count
print(f"More than 5: {more_than_5:,} / {total:,} ({more_than_5/total*100:.2f}%)")
print(f"More than 10: {more_than_10:,} / {total:,} ({more_than_10/total*100:.2f}%)")
print()


print("Changes:")
print(f"Total changes: {total_changes:,}")
for change, count in change_counter.most_common(200):
    print(change, f"{count:,} / {total_changes:,} ({count/total_changes*100:.2f}%)")
            