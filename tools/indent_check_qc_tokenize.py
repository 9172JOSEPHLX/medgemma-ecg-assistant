### tools/indent_check_qc_tokenize.py  ### Feb 13th, 2026. 12H50

import tokenize

p = r"src\medgem_poc\qc.py"

# Collect line numbers that belong to STRING tokens (docstrings, multiline strings, etc.)
string_lines = set()
comment_lines = set()

with open(p, "rb") as f:
    for tok in tokenize.tokenize(f.readline):
        ttype = tok.type
        (srow, scol) = tok.start
        (erow, ecol) = tok.end

        if ttype == tokenize.STRING:
            for ln in range(srow, erow + 1):
                string_lines.add(ln)
        elif ttype == tokenize.COMMENT:
            comment_lines.add(srow)

lines = open(p, "r", encoding="utf-8", errors="replace").read().splitlines()

tabs = []
bad4 = []

for i, s in enumerate(lines, start=1):
    if "\t" in s:
        tabs.append(i)

    # Skip docstrings / any string literal lines
    if i in string_lines:
        continue

    st = s.strip()
    if not st:
        continue

    # Skip pure comment lines
    if st.startswith("#") or i in comment_lines:
        continue

    ind = len(s) - len(s.lstrip(" "))
    if ind % 4 != 0:
        bad4.append((i, ind, s[:140]))

print("FILE", p)
print("TAB_LINES", len(tabs))
if tabs:
    print("TAB at:", tabs[:60])
print("INDENT_NOT_MULTIPLE_OF_4 (code only)", len(bad4))
for i, ind, ss in bad4[:200]:
    print(f"{i}: indent={ind} | {ss}")

# Terminus