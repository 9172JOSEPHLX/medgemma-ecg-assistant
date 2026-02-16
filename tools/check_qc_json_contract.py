import json
import sys
from datetime import datetime

VALID_STATUS = {"PASS", "WARN", "FAIL"}

def die(msg: str, code: int = 2):
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(code)

def require(obj, key):
    if key not in obj:
        die(f"missing top-level key: {key}")

def is_iso8601(s: str) -> bool:
    # accept "Z" suffix
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        datetime.fromisoformat(s)
        return True
    except Exception:
        return False

def main(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)  # strict: single JSON object only
    except Exception as e:
        die(f"cannot parse JSON: {e}")

    if not isinstance(data, dict):
        die("root must be a JSON object")

    # Required top-level keys
    for k in ("schema","generated_at","input","pipeline","status","reasons","warnings","metrics"):
        require(data, k)

    if data["schema"] != "qc.v1":
        die("schema must be 'qc.v1'")

    if not isinstance(data["generated_at"], str) or not is_iso8601(data["generated_at"]):
        die("generated_at must be ISO-8601 string (e.g. 2026-02-03T19:42:10Z)")

    if data["status"] not in VALID_STATUS:
        die(f"status must be one of {sorted(VALID_STATUS)}")

    # reasons
    reasons = data["reasons"]
    if not isinstance(reasons, list) or not all(isinstance(x, str) and x for x in reasons):
        die("reasons must be a list of non-empty strings")

    # warnings
    warnings = data["warnings"]
    if not isinstance(warnings, list):
        die("warnings must be a list")
    for i, w in enumerate(warnings):
        if not isinstance(w, dict):
            die(f"warnings[{i}] must be an object")
        if "code" not in w or "reason" not in w:
            die(f"warnings[{i}] must contain code + reason")
        if not isinstance(w["code"], str) or not w["code"]:
            die(f"warnings[{i}].code must be a non-empty string")
        if not isinstance(w["reason"], str) or not w["reason"]:
            die(f"warnings[{i}].reason must be a non-empty string")
        if w["code"] not in reasons:
            die(f"warning.code '{w['code']}' missing from reasons[]")

    # input/pipeline/metrics basic shape checks (non bloquant mais safe)
    if not isinstance(data["input"], dict):
        die("input must be an object")
    if not isinstance(data["pipeline"], dict):
        die("pipeline must be an object")
    if not isinstance(data["metrics"], dict):
        die("metrics must be an object")

    print("OK: qc.v1 contract validated.")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/check_qc_json_contract.py path/to/qc_report.json", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
