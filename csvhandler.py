import csv
import re

INPUT_CSV = "validation.csv"
OUTPUT_TXT = "data/char/val.txt"

_ws = re.compile(r"\s+")
_apostrophe_spaces = re.compile(r"\s*'\s*")

def normalize_text(t: str) -> str:
    if t is None:
        return ""

    t = t.strip()

    # normalize “smart” quotes/apostrophes to plain ones
    t = (t.replace("\u2019", "'")   # ’
           .replace("\u2018", "'")  # ‘
           .replace("\u201C", '"')  # “
           .replace("\u201D", '"')  # ”
           .replace("\u00A0", " ")) # nbsp

    # fix spaced-apostrophe tokens: don ' t -> don't, that ' s -> that's, Let ' s -> Let's
    t = _apostrophe_spaces.sub("'", t)

    # optional: normalize weird spacing around punctuation a bit
    t = t.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" ;", ";").replace(" :", ":")

    # collapse whitespace
    t = _ws.sub(" ", t).strip()

    return t

def extract_turns(dialog_cell: str):
    s = (dialog_cell or "").strip()
    if not s:
        return []

    s = s.replace('""', '"')

    l = s.find("[")
    r = s.rfind("]")
    if l != -1 and r != -1 and r > l:
        s = s[l+1:r]

    turns = []
    buf = []
    in_quote = False
    q = None

    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if not in_quote and ch in ("'", '"'):
            in_quote = True
            q = ch
            buf = []
            i += 1
            continue

        if in_quote and ch == q:
            if q == "'":
                j = i + 1
                while j < n and s[j].isspace():
                    j += 1
                if j >= n or s[j] in ("]", "'", '"'):
                    text = "".join(buf).strip()
                    if text:
                        turns.append(text)
                    in_quote = False
                    q = None
                    buf = []
                    i += 1
                    continue
                else:
                    buf.append("'")
                    i += 1
                    continue
            else:
                text = "".join(buf).strip()
                if text:
                    turns.append(text)
                in_quote = False
                q = None
                buf = []
                i += 1
                continue

        if in_quote:
            buf.append(ch)

        i += 1

    return turns

with open(INPUT_CSV, newline="", encoding="utf-8") as f_in, open(OUTPUT_TXT, "w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)

    for row in reader:
        turns = extract_turns(row.get("dialog", ""))
        if not turns:
            continue

        turns = [normalize_text(t) for t in turns]

        for i, utt in enumerate(turns):
            role = "User" if i % 2 == 0 else "Assistant"
            f_out.write(f"{role}: {utt}\n")
        f_out.write("\n")

print("done.")