import json
import argparse
from Bio.Align import PairwiseAligner
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colorist import Color

def read_book(book_id):
    with open(f'../ecco-paper-data/final_ecco_tcp_joined_books.jsonl') as f:
        for line in f:
            book = json.loads(line)
            if book['book_id'] == book_id:
                return book
    return None

def read_results(book_id):
    with open(f'eval_results.jsonl') as f:
        for line in f:
            result = json.loads(line)
            if result['book_id'] == book_id:
                return result
    return None

def preprocess(text):
    # remove whitespace
    text = " ".join(text.strip().split())
    return unicodedata.normalize("NFKC", text.casefold())

def init_aligner():
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -1
    aligner.mismatch_score = -0.5
    return aligner

def align_books(aligner, text1, text2):
    alignments = aligner.align(text1, text2)
    alignment = alignments[0]
    print(alignment)


def convert2fig(np_matrix):

    # Define a custom colormap
    cmap = ListedColormap(['white', 'red', 'black'])

    # Create the figure and axis
    w, h = np_matrix.shape
    pixel_per_cell = 10  # how many pixels (n*n) will represent each element in the array
    dpi = 20 # "Dots Per Inch", how many pixels are displayed per inch of physical space (image resolution)

    fig_width = w * pixel_per_cell / dpi  # Convert to inches (assuming 100 dpi)
    fig_height = h * pixel_per_cell / dpi

    # Create the figure with the calculated size and dpi
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)


    # Display the image
    plt.imshow(np_matrix, cmap=cmap, origin='upper', interpolation='nearest')

    # Show the plot
    plt.title("Custom Colormap Visualization")
    plt.savefig("test.png")

def create_alignment_matrix(alignment_array, text1, text2):
    # create an empty matrix od correct size
    matrix = np.zeros((len(text1), len(text2)))
    text1_alignments = alignment_array[0]
    text2_alignments = alignment_array[1]
    assert len(text1_alignments) == len(text2_alignments)
    for a1, a2 in zip(text1_alignments, text2_alignments):
        a1_start, a1_end = a1
        a2_start, a2_end = a2
        assert a1_end - a1_start == a2_end - a2_start # equal length
        l = []
        for i in range(0, a1_end - a1_start): # iterate over alignment
            text1_char = text1[a1_start+i]
            text2_char = text2[a2_start+i]
            l.append(text1_char == text2_char)
            if text1_char == text2_char:
                matrix[a1_start+i, a2_start+i] = 1.0
            else:
                matrix[a1_start+i, a2_start+i] = 0.5

        #print(a1, a2)
        #print(text1[a1[0]:a1[1]], "|||", text2[a2[0]:a2[1]])
        #print(l)
        #print()

    print("ecco input:", text1[1382:1936])
    print("ecco corrected:", text2[1344:1827])

    print(matrix.shape)
    convert2fig(matrix[1382:1936,1344:1827])


def color_string(string, color_list):
    colored_chars = [f"{Color.GREEN}{c}{Color.OFF}" if color==True else f"{Color.RED}{c}{Color.OFF}" for c, color in zip(string, color_list)]
    return "".join(colored_chars)


def visualize_1d(input, output, alignment_array):

    input_alignments = alignment_array[0]
    output_alignments = alignment_array[1]
    assert len(input_alignments) == len(output_alignments)
    output_char_alignments = [False]*len(output) # initialize with False

    for a1, a2 in zip(input_alignments, output_alignments):
        a1_start, a1_end = a1
        a2_start, a2_end = a2
        assert a1_end - a1_start == a2_end - a2_start # equal length
        for i in range(0, a1_end - a1_start): # iterate over alignment
            input_char = input[a1_start+i]
            output_char = output[a2_start+i]
            if input_char == output_char:
                output_char_alignments[a2_start+i] = True

    print(color_string(output, output_char_alignments))
    
 



def main(args):

    book = read_book(args.book_id)
    if not book:
        print(f"Book {args.book_id} not found")
        exit()
    print("FULL BOOK\n###########")
    print(book['tcp reference'])
    print("####################################")

    #results = read_results(args.book_id)
    #if not results:
    #    print(f"Results for book {args.book_id} not found")
    #    exit()

    aligner = init_aligner()
    orig = preprocess(book['ecco input'])
    corr = preprocess(book['ecco corrected postprocessed'])
    ref = preprocess(book['tcp reference'])

    #print("ecco input:", " ".join(book['ecco input'].strip().split()), "\n")
    #print("ecco corrected:", " ".join(book['ecco corrected'].strip().split()), "\n")


    #print("Top: ecco input")
    #print("Bottom: ecco corrected")
    #align_books(aligner, orig, corr)

    #print("\n\n")

    #print("Top: ecco corrected")
    #print("Bottom: tcp reference")
    #align_books(aligner, corr, preprocess(book['tcp reference']))

    #a = create_alignment_matrix(aligner.align(orig, corr)[0].aligned, orig, corr)

    print("Original ECCO text (green = present in corrected, red = deleted or changed):")
    visualize_1d(corr, orig, aligner.align(corr, orig)[0].aligned)
    print()
    print("Corrected ECCO text (green = present in original, red = added or changed):")
    visualize_1d(orig, corr, aligner.align(orig, corr)[0].aligned)
    print()
    print("Reference TCP text (green = present in corrected, red = missing or does not match):")
    visualize_1d(corr, ref, aligner.align(corr, ref)[0].aligned)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Visualize a book')
    parser.add_argument('--book_id', type=str, help='The book to visualize')

    args = parser.parse_args()

    main(args)
