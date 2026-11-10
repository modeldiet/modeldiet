# Profile a model. Understand its inference cost.

Model Diet is a Python library for measuring inference characteristics of an already-fitted model. It returns a structured `ProfileResult` with metric definitions and can export the profile as JSON or Markdown.

The default profile measures **single-row prediction latency only**. It does not measure accuracy or determine whether predictions are correct.

## Start here

1. [Install Model Diet and run the quick start](quickstart.md).
2. Read the [application guide](application.md) for supported models, inputs, exports, and error handling.
3. Use the [measurement guide](measurements.md) to understand the meaning and limits of each metric.
4. Create a tabbed, interactive report with the [visualization guide](visualization.md).
5. Look up public functions and result classes in the [API reference](api.md).

To preview the documentation locally or configure GitHub Pages, see [Host the docs](hosting.md).

## At a glance

```python
from modeldiet import profile

result = profile(model=fitted_model, X=representative_inputs)
result.to_json("profile.json")
```

The model must have a callable `predict(X)` method. Model Diet supplies each row as a one-row batch.

## Install

Install the core package from production PyPI:

```bash
python -m pip install modeldiet
```

The scikit-learn example requires the optional demo extra:

```bash
python -m pip install "modeldiet[demo]"
```

See the repository [README](https://github.com/modeldiet/modeldiet/blob/master/README.md) for the project overview and the [contribution guide](https://github.com/modeldiet/modeldiet/blob/master/CONTRIBUTING.md) for development setup.
