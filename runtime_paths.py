"""All machine-specific dependencies are configured outside the source code."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def xvlm_root():
    base = Path(os.environ.get('NEURIPS_ROOT', str(ROOT.parent / 'WeatherPrompt'))).expanduser()
    return Path(os.environ.get('XVLM_ROOT', str(base / 'XVLM/X-VLM-master'))).expanduser().resolve()

def bert_model():
    return os.environ.get('BERT_MODEL', 'bert-base-uncased')

def train_captions():
    return os.environ.get('TRAIN_CAPTIONS', '')

def configure_imports():
    base = Path(os.environ.get('NEURIPS_ROOT', str(ROOT.parent / 'WeatherPrompt'))).expanduser().resolve()
    # Local IJCV modules win over same-named upstream modules.
    for value in [str(ROOT), str(xvlm_root()), str(base)]:
        if value in sys.path:
            sys.path.remove(value)
    sys.path[:0] = [str(ROOT), str(xvlm_root()), str(base)]
