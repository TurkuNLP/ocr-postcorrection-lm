import json
import gzip
import argparse
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool


def read_tcp_books(fname):
    # list of tcp filenames, tcp data is on page level, combines pages into books
    books = {} # key: book_id, value: dictionary, where key: page number, value: {"input": ..., "output": ...}

    if fname.endswith(".gz"):
        f = gzip.open(fname, 'rt', encoding="utf-8")
    else:
        f = open(fname, 'rt', encoding="utf-8")
        
    for line in f:
        page = json.loads(line)
        book_id, page_number = page["doc_id"].split("_", 1)
        page_number = int(page_number)
        if book_id not in books:
            books[book_id] = {}
        assert page_number not in books[book_id]
        books[book_id][page_number] = {"input": page["input"], "output": page["output"]}
    f.close()
    print(f"TCP books read: {len(books)}")

    keys = list(books.keys())
    for book_id in keys:
        pages = books[book_id].keys()
        if len(pages) != max(pages):
            print(f"Book {book_id} has missing pages, deleting it. Missing pages: {[i for i in range(1, len(pages)+1) if i not in pages]}")
            del books[book_id]
    print(f"TCP books remaining: {len(books)}")
    return books


def join_texts(books):
    joined_books = {}
    for book_id, pages in books.items():
        sorted_pages = sorted(pages.keys()) # ensure correct page order
        input = "\n\n".join([pages[page_number]["input"] for page_number in sorted_pages if pages[page_number]["input"].strip().lower() != "blank"])
        # same for output
        output = "\n\n".join([pages[page_number]["output"] for page_number in sorted_pages if pages[page_number]["output"].strip().lower() != "blank"])
        joined_books[book_id] = {"tcp input": input, "tcp reference": output, "book_id": book_id}
    return joined_books


def read_ecco_books_in_parallel(counter, books_to_keep, ecco_path, combine_segments):
    # list of ecco filenames, ecco data is on book level
    my_files = glob.glob(os.path.join(ecco_path, f"{counter}.jsonl"))
    books = {} # key: book_id, value: {"input": ..., "output": ...}
    assert len(my_files) == 1
    fname = my_files[0]
    with open(fname, 'rt') as f:
        for line in f:
            book = json.loads(line)
            url = book["url"]
            book_id = url.split("/")[-1].replace(".txt", "")
            if book_id not in books_to_keep:
                continue
            original_text = book["text-orig"]
            #corrected_text = " ".join(book["corrections"]) ## TODO
            corrected_text = book["answer"]
            corrected_text_postprocessed = book["post-processed-answer"]
            if combine_segments:
                original_text = " ".join(original_text)
                corrected_text = " ".join(corrected_text)
                corrected_text_postprocessed = " ".join(corrected_text_postprocessed)
            books[book_id] = {"ecco input": original_text, "ecco corrected": corrected_text, "ecco corrected postprocessed": corrected_text_postprocessed, "book_id": book_id}
    return books



def read_data(tcp_file, ecco_path, parallel, combine_segments):
    # read tcp
    print("Reading TCP books")
    tcp_books = read_tcp_books(tcp_file)
    # join tcp pages into books
    tcp_books = join_texts(tcp_books)
    print(f"TCP done")
    
    # read ecco
    print(f"Reading ECCO books in parallel using {parallel} processes")
    ecco_corrected_books = {}
    pool = Pool(parallel)
    args = [(i, list(tcp_books.keys()), ecco_path, combine_segments) for i in range(200)]
    for books in tqdm(pool.starmap(read_ecco_books_in_parallel, args)):
        ecco_corrected_books.update(books)
    print("Number of ECCO books:", len(ecco_corrected_books))

    # 
    book_ids = set(tcp_books.keys()) & set(ecco_corrected_books.keys())
    book_ids = list(book_ids)
    print("Common books:", len(book_ids))

    # merge
    for book_id in book_ids:
        ecco_corrected_books[book_id].update(tcp_books[book_id])
    del tcp_books
    return [ecco_corrected_books[book_id] for book_id in book_ids]




   

def main(args):

    # read TCP books
    books = read_data(args.tcp_file, args.ecco_path, args.parallel, args.combine_segments)

    with open(args.output, "wt", encoding="utf-8") as f:
        for book in books:
            print(json.dumps(book, ensure_ascii=False), file=f)



    

if __name__ == "__main__":

    # read TCP books

    parser = argparse.ArgumentParser()
    parser.add_argument("--tcp-file", default="/scratch/project_2005072/siiri/en_unfiltered.jsonl", help="Path to TCP file (jsonl with page level data)")
    parser.add_argument("--ecco-path", default="/scratch/project_2005072/cassandra/ecco-ocr-run-out-assessment/postprocessed.v2", help="Path to ECCO data, directory with files from 0.jsonl to 199.jsonl.")
    parser.add_argument("--parallel", type=int, default=20, help="Number of parallel processes")
    parser.add_argument("--combine-segments", action="store_true", help="For each book, combine segments into one string.")
    parser.add_argument("--output", default="ecco_tcp_joined_books.jsonl", help="Path to output file")
    args = parser.parse_args()

    main(args)

    # python create_tcp_subset.py --tcp-file /scratch/project_2005072/siiri/en_unfiltered.jsonl --ecco-path /scratch/project_2005072/cassandra/ecco-ocr-run-out-assessment/postprocessed.v2 --parallel 20 --combine-segments --output final_ecco_tcp_joined_books.jsonl