# Random Projection on the KDD Cup 1999 Dataset

CSE558 Data Science assignment (IIIT-Delhi). We study how random projection (the Johnson–Lindenstrauss lemma) speeds up two tasks on the KDD Cup 1999 network-intrusion data while keeping their results close to the exact ones.

- **Task 1: JL for k-means clustering.** Project the data to `m` dimensions with a dense Gaussian matrix, cluster there, and compare the cost of the resulting centres with k-means on the full data.
- **Task 2: JL for linear regression.** Sketch the rows of the least-squares problem with a sparse CountSketch, solve the small problem, and compare its loss and running time with the exact solution. Also compare CountSketch with a dense Gaussian sketch.

## Repository contents

| File | Description |
|---|---|
| `Task_1.ipynb` | JL for k-means clustering (k = 15) |
| `Task_2.ipynb` | JL for least-squares regression with CountSketch |
| `DSc_Report.pdf` | Report with all tables, plots and answers |

## Data

Both notebooks load the default 10% version of KDD Cup 1999 through scikit-learn:

```python
from sklearn.datasets import fetch_kddcup99
kdd = fetch_kddcup99(subset=None, as_frame=True)
```

The data matrix `D` is built by one-hot encoding `protocol_type`, `service` and `flag`, converting the other features to float, dropping constant columns and rescaling each column to [0, 1]. The result has **n = 494,021 rows** and **d = 116 columns**.

## How to run

The notebooks were run on Google Colab. To run them locally:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook
```

Then open a notebook and run all cells from top to bottom. The dataset is downloaded automatically on the first run. Each notebook takes a few minutes; Task 1 is the slower one because it runs k-means many times.

All random seeds are fixed, so every number except the timings is reproducible.

## Main results

**Task 1 (k-means):**
- Clustering in only 15 to 25 dimensions gives centres whose cost on the full data is within about 4% of clustering `D` directly, and the whole computation is about 2.7 to 4.2 times faster.
- Averaging the original rows of each cluster gives far better centres than mapping back with the pseudo-inverse.

**Task 2 (least squares):**
- Below `m = rank(D) = 113` the sketched problem is fitted exactly and the solution is poor; above it the loss ratio falls steadily to about 1.02 at `m = 5000`.
- Sketch-and-solve is about 4 to 5 times faster than the exact solve.
- CountSketch and a dense Gaussian sketch give about the same quality, but CountSketch is about 3.7 times faster and never builds a dense matrix.

## Authors

- Malav Mehulkumar Solanki (2023306)
- Rishith Rohan Anamalla (2023081)
