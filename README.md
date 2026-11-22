# CSE558 Data Science Assignment


Malav Mehulkumar Solanki (2023306) · Rishith Rohan Anamalla (2023081)

This repository contains our code and report for a study of random projection (the Johnson–Lindenstrauss lemma) on the KDD Cup 1999 network-intrusion dataset:

- **Question 1:** JL for $k$-means clustering
- **Question 2:** JL for least-squares regression with a sparse CountSketch

| File | Description |
|---|---|
| `Task_1.ipynb` | Question 1: JL for $k$-means clustering ($k = 15$) |
| `Task_2.ipynb` | Question 2: JL for linear regression |
| `DSc_Report.pdf` | The submitted report (same content as this README) |
| `figs/` | Plots used below |

**How to run.** Open a notebook in Google Colab or Jupyter and run all cells from top to bottom. The dataset is downloaded automatically. Needs `numpy`, `pandas`, `matplotlib`, `seaborn` and `scikit-learn`. All random seeds are fixed, so every number except the timings is reproducible.

---

## 1. Data

The default 10% KDD Cup 1999 data was loaded with `fetch_kddcup99(subset=None, as_frame=True)`. The data matrix $D$ was built by one-hot encoding `protocol_type`, `service` and `flag`, converting the other features to float, dropping constant columns, and rescaling each column to $[0,1]$.

| | |
|---|---|
| Number of rows $n$ | 494,021 |
| Number of columns $d$ | 116 |

---

## 2. Question 1: JL for $k$-means clustering ($k = 15$)

Every clustering uses `KMeans(n_clusters=15, n_init=5, random_state=67)`. The data is projected as $E = DA^\top$, where $A \in \mathbb{R}^{m\times d}$ has i.i.d. $N(0, 1/m)$ entries.

### (a) $k$-means on $D$

| | |
|---|---|
| cost$(D,B)$ | 53230.53 |
| Wall-clock time | 21.52 s |

This cost is the baseline for all later ratios.

### (b) $k$-means on $E$; cost of $\hat B$

$\hat B$ is formed by averaging the original rows of $D$ in each cluster found on $E$. There were 5 runs per $m$, with a fresh $A$ each time.

| $m$ | mean ratio | min ratio | max ratio |
|---:|---:|---:|---:|
| 5 | 1.1526 | 1.0452 | 1.2086 |
| 10 | 1.0978 | 1.0367 | 1.1570 |
| 15 | 1.0433 | 1.0131 | 1.0769 |
| 20 | 1.0385 | 1.0055 | 1.0602 |
| 25 | 1.0434 | 0.9730 | 1.1083 |

*cost$(D,\hat B)$ / cost$(D,B)$ over 5 runs.*

As $m$ increases the mean ratio decreases, from about 1.15 at $m=5$ to about 1.04 for $m \ge 15$, because the JL distortion $\varepsilon$ shrinks as the number of dimensions grows. One run at $m=25$ has a ratio below 1 (0.973). This is possible because $k$-means only finds a local optimum, so cost$(D,B)$ is not the minimum possible cost, and a clustering found on $E$ can happen to give a cheaper $\hat B$.

### (c) $\hat B$ versus $B^+$

$B^+$ maps each centre found on $E$ back to $\mathbb{R}^d$ with the pseudo-inverse $A^+$.

| $m$ | mean cost$(D,\hat B)$/cost$(D,B)$ | mean cost$(D,B^+)$/cost$(D,B)$ |
|---:|---:|---:|
| 5 | 1.1526 | 80.98 |
| 10 | 1.0978 | 77.65 |
| 15 | 1.0433 | 72.66 |
| 20 | 1.0385 | 69.11 |
| 25 | 1.0434 | 65.33 |

$B^+$ is far worse: its cost is 65 to 81 times the cost of $B$. The pseudo-inverse maps each centre into the $m$-dimensional row space of $A$, a small subspace of $\mathbb{R}^{116}$, so the mapped centres cannot lie near the data. $B^+$ improves as $m$ grows, but is still very poor. $\hat B$ averages the original rows, so it lies where the data lies. **We would use $\hat B$.**

### (d) Running time

| $m$ | mean time (s) | fraction of time for $D$ (21.52 s) |
|---:|---:|---:|
| 5 | 5.15 | 0.24 |
| 10 | 5.82 | 0.27 |
| 15 | 7.95 | 0.37 |
| 20 | 7.72 | 0.36 |
| 25 | 8.05 | 0.37 |

![Q1(d): time against m](figs/q1d.png)

*Time to compute $E$ plus $k$-means on $E$ (mean of 5 runs). The dashed line is $k$-means on $D$ from (a).*

Projecting first is about 2.7 to 4.2 times faster than clustering $D$. The time grows with $m$ because each $k$-means iteration costs about $O(nmk)$.

### (e) Distance preservation and cluster cost

(i) For one draw of $A$: $\lVert E_i - E_l\rVert^2 / \lVert D_i - D_l\rVert^2$ over $10^4$ fixed random pairs. (ii) For 20 draws of $A$: cost$_E(\mathcal{C}_D)$ / cost$_D(\mathcal{C}_D)$ for the clusters from (a).

| $m$ | (i) mean | (i) std | (ii) mean | (ii) std |
|---:|---:|---:|---:|---:|
| 5 | 0.620 | 0.399 | 1.032 | 0.123 |
| 10 | 1.023 | 0.378 | 1.002 | 0.105 |
| 15 | 1.000 | 0.299 | 0.995 | 0.078 |
| 20 | 1.168 | 0.329 | 1.012 | 0.060 |
| 25 | 0.893 | 0.261 | 1.016 | 0.077 |

![Q1(e): box plots](figs/q1e.png)

The cluster-cost ratio (ii) has the smaller spread for every $m$. It is a sum over all 494,021 points, so the errors of individual distances average out, while (i) looks at single pairs. The spread of both decreases overall as $m$ grows.

<details>
<summary><b>All runs of Q1 (b), (c), (d)</b> (ratios relative to cost$(D,B)$ = 53230.53)</summary>

| $m$ | run | cost$(D,\hat B)$ | ratio $\hat B$ | ratio $B^+$ | time (s) |
|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 61903.2 | 1.1629 | 81.72 | 5.26 |
| 5 | 2 | 63530.9 | 1.1935 | 80.97 | 5.15 |
| 5 | 3 | 55635.3 | 1.0452 | 79.46 | 4.13 |
| 5 | 4 | 64333.8 | 1.2086 | 81.28 | 7.77 |
| 5 | 5 | 61364.6 | 1.1528 | 81.45 | 3.46 |
| 10 | 1 | 55838.7 | 1.0490 | 78.28 | 4.19 |
| 10 | 2 | 61590.1 | 1.1570 | 79.18 | 5.32 |
| 10 | 3 | 58557.6 | 1.1001 | 75.91 | 5.07 |
| 10 | 4 | 55182.2 | 1.0367 | 76.22 | 8.07 |
| 10 | 5 | 61014.7 | 1.1462 | 78.66 | 6.47 |
| 15 | 1 | 53927.8 | 1.0131 | 73.52 | 9.36 |
| 15 | 2 | 54974.5 | 1.0328 | 74.39 | 6.59 |
| 15 | 3 | 55517.0 | 1.0430 | 71.24 | 6.87 |
| 15 | 4 | 57324.7 | 1.0769 | 70.90 | 6.31 |
| 15 | 5 | 55937.3 | 1.0509 | 73.24 | 10.61 |
| 20 | 1 | 53521.7 | 1.0055 | 67.55 | 8.61 |
| 20 | 2 | 56254.6 | 1.0568 | 68.87 | 7.42 |
| 20 | 3 | 55560.5 | 1.0438 | 65.83 | 9.24 |
| 20 | 4 | 56435.9 | 1.0602 | 74.29 | 7.90 |
| 20 | 5 | 54621.0 | 1.0261 | 69.02 | 5.42 |
| 25 | 1 | 58992.8 | 1.1083 | 67.89 | 8.03 |
| 25 | 2 | 51792.8 | 0.9730 | 65.23 | 6.89 |
| 25 | 3 | 55314.8 | 1.0392 | 63.49 | 8.87 |
| 25 | 4 | 54185.6 | 1.0179 | 66.36 | 6.60 |
| 25 | 5 | 57409.4 | 1.0785 | 63.69 | 9.86 |

</details>

---

## 3. Question 2: JL for linear regression

$y_i = 1$ if connection $i$ is an attack and $0$ otherwise; 80.3% of the connections are attacks. The weights $b$ minimise $\mathrm{Loss}(D,y,a) = \lVert Da - y\rVert_2^2$ (no intercept). Every least-squares problem is solved with `numpy.linalg.lstsq`.

The rows are sketched with a **CountSketch**: each row $i$ gets a random bucket $h(i) \in \{1,\dots,m\}$ and a random sign $\sigma(i) \in \{-1,+1\}$, and is added with that sign to row $h(i)$ of $SD$ using `numpy.add.at`. $S$ is never stored as a dense matrix. $\tilde a$ minimises $\lVert SDa - Sy\rVert_2^2$.

### (a) Full least-squares solution

| | |
|---|---|
| Loss$(D,y,b)$ | 2949.789 |
| Time to compute $b$ | 10.48 s |
| rank$(D)$ | 113 (with $d = 116$) |

### (b) Sketched solutions for different $m$

There were 5 CountSketches per $m$, with a fresh sketch each time.

| $m$ | mean ratio | min ratio | max ratio | mean sketched loss |
|---:|---:|---:|---:|---:|
| 50 | 3.3936 | 2.6032 | 4.7491 | $4.1\times10^{-24}$ |
| 100 | 12.7933 | 9.1467 | 18.6676 | $3.2\times10^{-23}$ |
| 200 | 2.5781 | 2.3274 | 3.0404 | 1251.34 |
| 500 | 1.3250 | 1.2463 | 1.3830 | 2334.14 |
| 1000 | 1.1273 | 1.1178 | 1.1405 | 2535.61 |
| 2000 | 1.0618 | 1.0484 | 1.0721 | 2794.41 |
| 5000 | 1.0218 | 1.0188 | 1.0261 | 2919.49 |

*Loss$(D,y,\tilde a)$ / Loss$(D,y,b)$ over 5 sketches per $m$, and the mean sketched loss $\lVert SD\tilde a - Sy\rVert^2$.*

![Q2(b): loss ratio against m](figs/q2b.png)

*Mean ratio against $m$ (log–log). The error bars are the smallest and largest of the 5 ratios. The dotted line is $m = \mathrm{rank}(D) = 113$.*

**Below $m = \mathrm{rank}(D) = 113$** ($m = 50, 100$): the sketched loss is essentially 0, because the small system has fewer equations than unknowns and is fitted exactly, but the ratio is large (3.39 and 12.79). **Above 113** ($m \ge 200$): the sketched loss is no longer 0; on average it is below Loss$(D,y,b)$ = 2949.79 and rises towards it as $m$ grows. The ratio falls steadily, to 1.02 at $m = 5000$, and the spread between the 5 runs shrinks.

<details>
<summary><b>All runs of Q2(b)</b></summary>

| $m$ | run | Loss$(D,y,\tilde a)$ | ratio | sketched loss |
|---:|---:|---:|---:|---:|
| 50 | 1 | 14008.74 | 4.7491 | $5.1\times10^{-24}$ |
| 50 | 2 | 10019.98 | 3.3968 | $5.1\times10^{-24}$ |
| 50 | 3 | 7678.94 | 2.6032 | $6.4\times10^{-25}$ |
| 50 | 4 | 8847.66 | 2.9994 | $4.0\times10^{-24}$ |
| 50 | 5 | 9496.05 | 3.2192 | $5.5\times10^{-24}$ |
| 100 | 1 | 26980.78 | 9.1467 | $1.4\times10^{-23}$ |
| 100 | 2 | 32602.99 | 11.0527 | $2.4\times10^{-23}$ |
| 100 | 3 | 55065.55 | 18.6676 | $4.8\times10^{-23}$ |
| 100 | 4 | 40439.07 | 13.7091 | $3.4\times10^{-23}$ |
| 100 | 5 | 33599.50 | 11.3905 | $4.2\times10^{-23}$ |
| 200 | 1 | 8968.49 | 3.0404 | 1149.41 |
| 200 | 2 | 6865.20 | 2.3274 | 1466.46 |
| 200 | 3 | 7189.71 | 2.4374 | 1370.58 |
| 200 | 4 | 6894.64 | 2.3373 | 1177.25 |
| 200 | 5 | 8106.00 | 2.7480 | 1092.98 |
| 500 | 1 | 3918.86 | 1.3285 | 2159.58 |
| 500 | 2 | 3973.90 | 1.3472 | 2358.40 |
| 500 | 3 | 3676.34 | 1.2463 | 2417.17 |
| 500 | 4 | 4079.49 | 1.3830 | 2276.84 |
| 500 | 5 | 3893.53 | 1.3199 | 2458.71 |
| 1000 | 1 | 3311.41 | 1.1226 | 2629.60 |
| 1000 | 2 | 3297.24 | 1.1178 | 2353.97 |
| 1000 | 3 | 3327.78 | 1.1281 | 2622.67 |
| 1000 | 4 | 3325.59 | 1.1274 | 2573.88 |
| 1000 | 5 | 3364.13 | 1.1405 | 2497.94 |
| 2000 | 1 | 3092.65 | 1.0484 | 2680.12 |
| 2000 | 2 | 3141.95 | 1.0651 | 2933.00 |
| 2000 | 3 | 3162.47 | 1.0721 | 2822.02 |
| 2000 | 4 | 3117.80 | 1.0570 | 2850.53 |
| 2000 | 5 | 3146.22 | 1.0666 | 2686.38 |
| 5000 | 1 | 3015.30 | 1.0222 | 2894.97 |
| 5000 | 2 | 3013.79 | 1.0217 | 2973.81 |
| 5000 | 3 | 3005.18 | 1.0188 | 2845.11 |
| 5000 | 4 | 3009.41 | 1.0202 | 2939.85 |
| 5000 | 5 | 3026.77 | 1.0261 | 2943.73 |

</details>

### (c) Running time

This is the time to draw $S$, form $SD$ and $Sy$, and compute $\tilde a$, averaged over the 5 runs of (b).

| $m$ | mean time (s) | speed-up over exact (10.48 s) |
|---:|---:|---:|
| 50 | 2.19 | 4.80× |
| 100 | 2.37 | 4.42× |
| 200 | 2.20 | 4.78× |
| 500 | 2.08 | 5.05× |
| 1000 | 2.25 | 4.65× |
| 2000 | 2.30 | 4.56× |
| 5000 | 2.45 | 4.28× |

The sketch-and-solve time is about 2.1 to 2.4 s for every $m$, against 10.48 s for the exact solve in (a), so it is about 4.3 to 5 times faster. The time hardly changes with $m$ because it is dominated by the single pass over all $n$ rows needed to form $SD$; solving the small $m \times d$ problem is cheap even at $m = 5000$.

### (d) Preservation of the residual's length

Here $v = Db - y$. For each $m$, 100 independent CountSketches were drawn and $\lVert Sv\rVert^2 / \lVert v\rVert^2$ computed.

| $m$ | mean | standard deviation |
|---:|---:|---:|
| 50 | 0.9891 | 0.1876 |
| 500 | 0.9965 | 0.0676 |
| 5000 | 1.0000 | 0.0214 |

![Q2(d): histograms](figs/q2d.png)

For every $m$ the mean is close to 1, so a random sketch keeps the squared length of $v$ correct on average. The spread shrinks as $m$ grows: the standard deviation is 0.188, 0.068 and 0.021 for $m = 50, 500, 5000$, and the histograms become narrower.

### (e) CountSketch versus dense Gaussian sketch

50,000 rows were chosen once at random (fixed seed), giving $D'$ and $y'$, with Loss$(D',y',b')$ = 261.14. For $m = 500$ we sketch with (i) a CountSketch and (ii) a dense $S \in \mathbb{R}^{500\times 50000}$ with i.i.d. $N(0,1/m)$ entries. The time is that of drawing $S$ and forming $SD'$ and $Sy'$.

| Sketch | mean time (s) | mean Loss$(D',y',\tilde a)$/Loss$(D',y',b')$ |
|---|---:|---:|
| CountSketch | 0.241 | 1.280 |
| Dense Gaussian | 0.894 | 1.285 |

| run | CountSketch time (s) | CountSketch ratio | Dense time (s) | Dense ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.307 | 1.2959 | 1.166 | 1.2519 |
| 2 | 0.279 | 1.2150 | 1.102 | 1.3287 |
| 3 | 0.263 | 1.2626 | 0.733 | 1.2740 |
| 4 | 0.175 | 1.3416 | 0.751 | 1.2978 |
| 5 | 0.182 | 1.2843 | 0.719 | 1.2707 |

**We would use CountSketch on the full $D$.** Both sketches give almost the same ratio (1.280 against 1.285), but CountSketch is about 3.7 times faster to form (0.24 s against 0.89 s). It also needs only one pass over the rows and never builds a dense $500 \times 494{,}021$ matrix, which would take about 2 GB in double precision.

---

## Random seeds

| Part | Seed |
|---|---|
| Q1: all $k$-means runs | `random_state=67` |
| Q1 (b)–(d): matrices $A$ | `default_rng((i+1)*100)`, one generator per $m$ |
| Q1 (e): pairs and 20 draws of $A$ | `default_rng(67)` |
| Q2 (b): CountSketches | `default_rng(m)` |
| Q2 (d): 100 CountSketches | `default_rng(10000 + m)` |
| Q2 (e): subset and sketches | `default_rng(67)` |

## References

- CSE558 lecture notes: Lecture 19 (Johnson–Lindenstrauss), Lecture 20 ($k$-means, SVD), Lecture 22 (least squares, subspace embeddings, sparse sketches)
- [sklearn.datasets.fetch_kddcup99](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)
- [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
- [numpy.add.at](https://numpy.org/doc/stable/reference/generated/numpy.add.at.html)
