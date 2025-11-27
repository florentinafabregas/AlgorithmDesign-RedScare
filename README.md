# AlgorithmDesign-RedScare

This repository contains our implementations for the *Red Scare* graph problems.  
All solver scripts are located in `code/`, and all input instances are in `red-scare/data/`.

## How to Reproduce All Results

After cloning the repository, run:

```bash
cd code
bash run_all_and_merge.sh
```

### What the script does

- Runs all solvers (`None.py`, `Some.py`, `Many.py`, `Few.py`, `Alternate.py`)  
  on every input graph in `../red-scare/data/`.

- Each solver writes a CSV file into `../results/`.

- `merge_results.py` combines these CSV files into a single `results.txt` file.

- The complete outputs will appear in the `results/` directory.
