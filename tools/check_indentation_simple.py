# tools/check_indentation_simple.py   Version du 02.02.2026 QC PY ENRICHI DES BLOCS “QC clinique/mesures”
import sys

def main(path: str) -> int:
    bad_tabs = []
    bad_mod4 = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            # ignore empty lines
            if not line.strip():
                continue
            # leading whitespace
            prefix = line[: len(line) - len(line.lstrip("\t "))]
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
