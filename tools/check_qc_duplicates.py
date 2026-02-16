import ast
import sys
from collections import Counter, defaultdict

def main(path: str) -> int:
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src, filename=path)

    seen_defs = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen_defs[node.name].append(node.lineno)
    dup_defs = {k: v for k, v in seen_defs.items() if len(v) > 1}

    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = ",".join(sorted([a.name for a in node.names]))
            imports.append(f"import:{names}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = ",".join(sorted([a.name for a in node.names]))
            imports.append(f"from:{mod}:{names}")
    c = Counter(imports)
    dup_imports = {k: v for k, v in c.items() if v > 1}

    assigns = defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigns[t.id].append(node.lineno)
    dup_consts = {k: v for k, v in assigns.items() if k.isupper() and len(v) > 1}

    ok = True
    if dup_defs:
        ok = False
        print("== DUPLICATE TOP-LEVEL DEFINITIONS ==")
        for name, lines in sorted(dup_defs.items()):
            print(f"{name}: lines {lines}")

    if dup_imports:
        ok = False
        print("\n== DUPLICATE IMPORT STATEMENTS ==")
        for k, v in sorted(dup_imports.items()):
            print(f"{k} -> x{v}")

    if dup_consts:
        ok = False
        print("\n== DUPLICATE TOP-LEVEL CONSTANT ASSIGNMENTS (ALL_CAPS) ==")
        for name, lines in sorted(dup_consts.items()):
            print(f"{name}: lines {lines}")

    if ok:
        print("OK: no duplicate defs/imports/constants detected.")
        return 0
    return 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/check_qc_duplicates.py path/to/qc.py")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
