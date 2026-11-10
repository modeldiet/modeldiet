# Quick start

This walkthrough installs the optional scikit-learn dependency, trains a small classifier, profiles inference latency, reads a result, and exports reports.

## 1. Install

Model Diet supports Python 3.11 or later. Install the demo extra from production PyPI:

```bash
python -m pip install "modeldiet[demo]"
```

## 2. Save and run this example

Save the following as `profile_model.py`, then run `python profile_model.py`:

```python
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from modeldiet import profile

rng = np.random.default_rng(42)
X = rng.random((100, 5))
y = rng.integers(0, 2, size=100)

model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X, y)

result = profile(model=model, X=X)
latency = result.measurements["latency"]
mean_ms = latency.metrics["mean_ms"].value
print(f"Mean single-row prediction latency: {mean_ms:.3f} ms")

output_dir = Path("profile-output")
output_dir.mkdir(exist_ok=True)
result.to_json(output_dir / "profile.json")
result.to_markdown(output_dir / "profile.md")
print(f"Reports written to {output_dir.resolve()}")
```

The generated data is only for demonstrating the API. For useful measurements, pass representative inputs with the exact feature layout and preprocessing your application uses.

## 3. Understand what ran

Because no `measurements` argument was supplied, Model Diet ran the default `LatencyMeasurement`. It warmed up with one prediction and then timed each of the 100 rows separately. The mean and percentile values describe those single-row predictions; they are not batch throughput or end-to-end request latency.

The profile does not contain accuracy. Model Diet does not receive labels or compare predicted values with expected values.

## 4. Read the output

Open `profile-output/profile.md` for a readable summary. The JSON file contains the same measurements in a structured format, including units and semantics for each metric. See [Measurements](measurements.md) for interpretation details and [API reference](api.md) for result fields.

To explore results from multiple models in an interactive dashboard, install
`modeldiet[viz]` and follow the [visualization guide](visualization.md). The
dashboard lets you focus on one model or compare two or three models using the
same metric rows and charts.
