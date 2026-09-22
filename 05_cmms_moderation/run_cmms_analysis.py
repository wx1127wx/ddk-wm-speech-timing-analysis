from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
PIPELINE = HERE.parents[1]
subprocess.run([
    sys.executable,
    str(HERE.parent / "cognitive_scale_analysis.py"),
    "--scale", "CMMS",
    "--analysis-root", str(PIPELINE),
    "--output-root", str(HERE.parent),
], check=True)
