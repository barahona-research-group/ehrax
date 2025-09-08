# Contributing

Contribution are very welcome. Types of contributions:

- Improving or expanding documentation.
- Code improvements, bug fixes, and new features.
- Contributing case studies as Jupyter notebooks.

# Contributing with pull requests

## Step 1: fork the repository

Fork on GitHub then clone your fork:

```bash
git clone https://github.com/your-username-here/ehrax.git
cd ehrax
pip install -e .
```

## Step 2: pre-commit hooks

Install pre-commit hooks that identify static typing errors (_via_ `pyright`) and formatting inconsistencies (_via_ `ruff`).

```bash
pip install pre-commit
pre-commit install
```

## Step 3: testing code changes

If you make changes to the code change/add to the tests as necessary. Then run the tests with:

```bash
pip install -r test/requirements.txt
pytest -n 8
```

## Step 4: testing documentation changes

If you make changes to the documentation, build the documentation with these commands:

```bash
pip install -e '.[docs]'
mkdocs serve
```

Then you can see a live rendered documentation in the link logged in the console.

## Last step: commit, push and pull request

The `git commit` command will invoke the `pre-commit` hooks. You have to fix any error to finish the commit process.

# The Codebase overview

## Project summary

This project implements a Python library for EHRs input+output operations, HDF serialisation, pipelined processing, for JAX+equinox compatible tasks, e.g. mainly ML.
Familiarise your self with [JAX](https://github.com/jax-ml/jax), [equinox](https://github.com/patrick-kidger/equinox) and [pytables](https://www.pytables.org) libraries, with focus on JAX PyTree and equinox PyTree manipulation routines.

## Project structure

* `docs/`: documentation supplementary files and notebook example studies
* `ehrax/*.py`: the library main source code
* `ehrax/example_datasets/`: API files to deal with common EHR datasets like MIMIC
* `ehrax/example_schemes/`: API files to create common clinical coding systems like ICD-9, ICD-10, and CCS
* `ehrax/resources/`: miscellaneous resources files
* `ehrax/testing/`: utilities for testing. Not testing code themselves.
* `test/`: tests are here.

The files in `ehrax/*.py` go into five categories:

1. `ehrax/utils.py` and `ehrax/base.py`: these are the core files that everything else depends on.
2. `ehrax/freezer.py` and `ehrax/coding_scheme.py`: `freezer.py` contains standard data structures that will be extensively utilised, while `coding_scheme.py` defines the data structures and the logic of clinical coding systems, outcome extractor, clinical codes mapping systems, and a context manager that contains all of these object types in the runtime.
3. `ehrax/dataset.py` and `ehrax/transformations.py`: `dataset.py` defines the Dataset interface that contains the table and configuration, while `transformations.py` contains essential and modular pipeline steps.
4. `ehrax/tvx_concepts.py`, `ehrax/tvx_ehr.py`, and `ehrax/tvx_transformation.py`: These are the hierarchical concept level representation of EHRs as opposed to tabular level. `tvx_concepts.py` defines the concepts starting from timestamped numerical representation of observables and interventions etc. and `tvx_ehr.py` is the EHRs representation of many patients, while `tvx_transformation.py` implements essential transformation steps that can be used as building blocks in processing pipelines.
5. `ehrax/_literals.py`: contains all literals used in the project so far in one place.

## Documentation

- `docs/` contains the rendered website, Jupyter notebooks, and static docs.
- `docs/index.md` is the landing page of the documentation site.
- `docs/notebooks/` holds Jupyter notebooks that demonstrate use‑cases.
-  `docs/faq.md` holds frequently asked questions.
-  `docs/why-ehrax.md` contains arguments and examples that convince users to adopt the library.

# Naming & Code Style

- File names and variables: **snake_case**.
- Max line length: 120 characters.
- Use 4 spaces for code indentation.
- All string literals that are reused across the code base live in `ehrax/_literals.py`.
