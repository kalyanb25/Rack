"""Stand-in for a Python-with-Unsloth: called as <python> train_unsloth.py --job J --out O."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fake_kaggle_outputs import write  # noqa: E402

args = sys.argv[1:]
job = Path(args[args.index("--job") + 1])
out = Path(args[args.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
write(out, job.parent)
print("[fake] trained")
