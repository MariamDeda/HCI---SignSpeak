"""Restore the SIGNS list in config.py from the trained model (models/meta.json).

Run once if config.py's SIGNS list got out of sync with your trained model:
    python restore_signs.py
It prints the old and new list and only edits the `SIGNS = [...]` line.
"""
import json
import re
from pathlib import Path

root = Path(__file__).parent
meta = json.loads((root / "models" / "meta.json").read_text())
signs = meta["signs"]
cfg_path = root / "config.py"
text = cfg_path.read_text()
m = re.search(r"^SIGNS\s*=\s*\[.*?\]", text, flags=re.S | re.M)
if not m:
    raise SystemExit("Could not find the SIGNS = [...] line in config.py")
new_line = "SIGNS = " + json.dumps(signs)
print("old:", m.group(0).replace("\n", " "))
print("new:", new_line)
cfg_path.write_text(text[:m.start()] + new_line + text[m.end():])
print("config.py updated. Check DISPLAY in config.py if you want nicer names for new signs.")
