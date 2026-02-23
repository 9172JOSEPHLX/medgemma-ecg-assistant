# tools/check_indentation_simple.py   Version du 02.02.2026 QC PY ENRICHI DES BLOCS “QC clinique/mesures”
# tools/check_indentation_simple.py — UPDATED (ignore triple-quoted blocks)
# Version Updated 23.022026  

import sys
from typing import Optional, Tuple


def _toggle_triple_state(line: str, state: Optional[str]) -> Tuple[Optional[str], bool]:
    """
    Best-effort detector for triple-quoted blocks (docstrings / multi-line strings).

    Returns (new_state, consumed_line):
      - state is None when not inside a triple-quoted block,
        otherwise it is one of: '\\"\\\"\\\"' or "\\'\\'\\'"
      - consumed_line indicates that this line contains a triple-quote delimiter and
        should be ignored for indentation checks (to reduce false positives).

    Notes:
      - If a line contains BOTH opening and closing triple quotes on the same line,
        we treat it as "consumed" and do not enter a block (state remains None).
      - This is a heuristic; it is intended for QC repo style, not a full Python parser.
    """
    # If currently inside a triple block, look for its closing delimiter
    if state is not None:
        if state in line:
            # close on this line
            return None, True
        # still inside block
        return state, True

    # Not inside: detect opening delimiter.
    # If the delimiter appears twice (open+close same line), consume and do not enter.
    for q in ('"""', "'''"):
        if q in line:
            if line.count(q) >= 2:
                return None, True  # one-liner triple quoted string/docstring
            return q, True  # enter block; consume this delimiter line
    return None, False


def main(path: str) -> int:
    bad_tabs = []
    bad_mod4 = []

    triple_state: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            # Detect and skip triple-quoted blocks (docstrings / multi-line strings)
            triple_state, consumed = _toggle_triple_state(line, triple_state)
            if consumed:
                continue

            # ignore empty lines
            if not line.strip():
                continue

            # leading whitespace
            prefix = line[: len(line) - len(line.lstrip("\t "))]

            # tabs in leading indentation
            if "\t" in prefix:
                bad_tabs.append(i)

            # spaces-only indent multiple of 4
            if prefix and ("\t" not in prefix):
                if (len(prefix) % 4) != 0:
                    bad_mod4.append((i, len(prefix)))

    if bad_tabs:
        print("== Tabs in leading indentation ==")
        print(", ".join(map(str, bad_tabs)))
    else:
        print("== Tabs in leading indentation ==\n(none)")

    if bad_mod4:
        print("\n== Indentation not multiple of 4 (spaces only) ==")
        for i, n in bad_mod4[:50]:
            print(f"{i}:INDENT%4!=0:{n}")
        if len(bad_mod4) > 50:
            print(f"... +{len(bad_mod4)-50} more")
    else:
        print("\n== Indentation not multiple of 4 (spaces only) ==\n(none)")

    return 1 if (bad_tabs or bad_mod4) else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/check_indentation_simple.py qc.py")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))