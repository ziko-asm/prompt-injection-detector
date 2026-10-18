import sys
from pathlib import Path

# make `import src...` work when pytest is run from the repository root
sys.path.insert(0, str(Path(__file__).parent))
