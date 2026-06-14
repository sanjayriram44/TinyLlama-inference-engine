import sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.loader import load_model
from engine.generator import naive_generate
from benchmarks.prompts import PROMPT

model, tok = load_model()
ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
_, t = naive_generate(model, ids, 128)
gaps = [ (t[i]-t[i-1])*1000 for i in range(1, len(t)) ]   # per-token ms
print("token  10:", round(gaps[10], 2), "ms")
print("token  60:", round(gaps[60], 2), "ms")
print("token 120:", round(gaps[120], 2), "ms")