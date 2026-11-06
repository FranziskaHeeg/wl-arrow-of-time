import json
from pathlib import Path

import numpy as np


def save_results(path, accuracies, **info):
    """Writes accuracies with their mean and standard deviation (plus any extra info) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**info, 'accuracies': list(accuracies),
               'mean': float(np.mean(accuracies)), 'std': float(np.std(accuracies))}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
