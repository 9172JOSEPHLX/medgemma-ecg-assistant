p=r"src\medgem_poc\qc.py"
lines=open(p,"r",encoding="utf-8",errors="replace").read().splitlines()

tabs=[]
bad4=[]
in_triple=None

def toggle(line, state):
    for q in ('"""',"'''"):
        if q in line:
            if state is None:
                return q
            elif state == q:
                return None
    return state

for i,s in enumerate(lines, start=1):
    if "\t" in s:
        tabs.append(i)

    prev=in_triple
    in_triple = toggle(s, in_triple)

    # ignore docstring blocks
    if prev is not None or in_triple is not None:
        continue

    st=s.strip()
    if (not st) or st.startswith("#"):
        continue

    ind=len(s)-len(s.lstrip(" "))
    if ind % 4 != 0:
        bad4.append((i, ind, s[:140]))

print("FILE", p)
print("TAB_LINES", len(tabs))
if tabs: print("TAB at:", tabs[:60])
print("INDENT_NOT_MULTIPLE_OF_4 (code only)", len(bad4))
for i,ind,ss in bad4[:120]:
    print(f"{i}: indent={ind} | {ss}")
