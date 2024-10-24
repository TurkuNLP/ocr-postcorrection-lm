import difflib
from rich.console import Console
from rich.text import Text

def highlight_differences_line_by_line(text1, text2):
    console = Console()

    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()
    
    max_lines = max(len(text1_lines), len(text2_lines))
    text1_lines += [""] * (max_lines - len(text1_lines))
    text2_lines += [""] * (max_lines - len(text2_lines))
    
    for line1, line2 in zip(text1_lines, text2_lines):
        matcher = difflib.SequenceMatcher(None, line1, line2)
        
        text1_highlighted = Text()
        text2_highlighted = Text()
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                text1_highlighted.append(line1[i1:i2])
                text2_highlighted.append(line2[j1:j2])
            elif tag == 'replace':
                text1_highlighted.append(line1[i1:i2], style="bold blue")
                text2_highlighted.append(line2[j1:j2], style="bold blue")
            elif tag == 'delete':
                text1_highlighted.append(line1[i1:i2], style="bold red")
                text2_highlighted.append(" " * (i2 - i1))
            elif tag == 'insert':
                text1_highlighted.append(" " * (j2 - j1)) 
                text2_highlighted.append(line2[j1:j2], style="bold green")

        console.print("Text 1: ", text1_highlighted)
        console.print("Text 2: ", text2_highlighted)
        console.print("\n")

text1 = """This is a simple example
to highlight differences
between two texts."""

text2 = """This is a simple sample
to highlight the differences
between two pieces of text."""

highlight_differences_line_by_line(text1, text2)
