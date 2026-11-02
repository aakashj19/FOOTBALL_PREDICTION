
import subprocess
import sys
from pathlib import Path

# Always run scripts from the project folder
PROJECT_DIR = Path(__file__).resolve().parent

steps = [
    ("Updating match data", "update_matches.py"),
    ("Rebuilding features", "recent_form.py"),
    ("Training model", "train_model.py"),
]

for message, script in steps:
    print(f"\n{'=' * 50}")
    print(message)
    print("=" * 50)

    script_path = PROJECT_DIR / script

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed.")
        sys.exit(result.returncode)

print("\nAll update steps completed successfully!")
