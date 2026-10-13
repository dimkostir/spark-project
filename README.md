# Parallel Vector Distance Join with LSH on Apache Spark

Approximate **Euclidean distance join** over high-dimensional vectors using **Locality-Sensitive Hashing (LSH)**, implemented in **PySpark**.

Course project for *Big Data Processing* — "Big Data and Analytics".

## Problem

Given two sets of `n`-dimensional vectors `R` and `S`, find every pair `(r, s)` with `r ∈ R`, `s ∈ S` whose Euclidean distance `d(r, s)` is below a user-defined threshold `θ`.

The exact solution compares every pair, which costs `O(|R| × |S|)`. That doesn't scale to millions of vectors. In a 10,000 × 10,000 test, the brute-force baseline was still running after more than 2 hours. This project trades a small loss in accuracy for a large gain in speed and scalability. The target is **recall ≥ 90%**.

## Approach: LSH for Euclidean distance

LSH uses hash functions that put nearby vectors in the same bucket with high probability. Only vectors that share a bucket are compared, so the exhaustive search is skipped.

Each hash function follows Datar et al. (2004), using p-stable distributions:

```
h(x) = ⌊(a · x + b) / w⌋
```

- `a`: random vector with entries drawn from `N(0, 1)`
- `b`: random number drawn uniformly from `[0, w]`
- `w`: bucket width

The index uses `L` independent hash tables, each with `k` hash functions. The `k` values in a table are concatenated into one composite key. Two vectors become a **candidate pair** when they share a bucket in **at least one** of the `L` tables.

| Parameter | Effect when increased |
|-----------|----------------------|
| `L` (hash tables) | Higher recall, more candidates, longer runtime |
| `k` (hash functions per table) | Fewer false positives, lower recall |
| `w` (bucket width) | Larger buckets, more candidates, higher recall |

## Algorithm (Spark pipeline)

1. **Load data**: read SIFT1M from `.fvecs` and build Spark DataFrames `(id, vector)`.
2. **Generate hash functions**: draw the random `a` and `b` in NumPy on the driver, then broadcast them to all workers.
3. **Compute hash keys**: a UDF applies `h(x)` for every table `l` and every hash function `j`, in parallel across vectors.
4. **Build composite keys**: join the `k` values of each table into one key such as `"3,7,2"`, which gives `L` keys per vector.
5. **Explode**: reshape each DataFrame to `(id, table_id, hash_key)` with `posexplode`.
6. **Generate candidates**: inner-join the exploded `R` and `S` on `(table_id, hash_key)`, then `distinct()`.
7. **Verify**: compute the exact Euclidean distance `d(r, s) = √Σ(rᵢ − sᵢ)²` for each candidate with a UDF.
8. **Filter**: keep only the pairs with `d(r, s) < θ`.

Recall is measured against a brute-force cross join on the same subset.

## Dataset

[**SIFT1M**](http://corpus-texmex.irisa.fr) (Jégou et al., 2011): 1,000,000 SIFT descriptors with 128 dimensions each. It is a standard benchmark for nearest-neighbor search.

The experiments use subsets of **2,000 × 2,000** vectors (`R` = vectors 0–1999, `S` = vectors 2000–3999). At this size, the brute-force baseline can still finish, so recall can be measured.

## Results

All experiments use `θ = 300`.

### Effect of `L` (k = 2, w = 300)

| L | Recall | LSH time (s) | Speedup vs baseline |
|---|--------|--------------|---------------------|
| 5  | 56.68% | 104 | 2.2× |
| 15 | 91.20% | 172 | 1.4× |
| 20 | 96.09% | 247 | 0.9× |

Recall rises sharply with `L`, but so does runtime. `L = 15` is the first value that reaches the 90% target.

### Effect of `w` (k = 3)

| L | w | Recall | LSH time (s) | Speedup |
|---|---|--------|--------------|---------|
| 30 | 300 | 83.80% | 112 | 2.1× |
| 25 | 450 | 97.76% | 174 | 1.4× |
| 20 | 600 | 99.50% | 212 | 1.1× |

Bucket width has a decisive effect on recall. It should be **at least θ**, so that buckets are large enough to hold most of the near neighbors.

### Configurations at ~91% recall

| k | L | w | Recall | LSH time (s) | Speedup |
|---|---|---|--------|--------------|---------|
| 2 | 15 | 300 | 91.20% | 172 | 1.4× |
| 3 | 25 | 450 | 91.04% | 124 | 2.0× |
| 4 | 20 | 600 | 91.35% | 112 | 2.1× |
| **4** | **30** | **450** | **91.16%** | **110** | **2.2×** |

At the same recall, raising `k` together with `w` gives smaller, more selective buckets. That means fewer candidates and faster verification. The best configuration, **k = 4, L = 30, w = 450**, is about 36% faster than the k = 2 setup.

## Conclusions

- With well-chosen parameters, LSH reaches the **> 90% recall** target.
- The best configuration found is **k = 4, L = 30, w = 450**: **91.16% recall** with a **2.2× speedup** over brute force.
- Parameter tuning is the main factor in result quality.
- Exact joins are impractical at scale: brute force did not finish 10,000 × 10,000 in 2 hours. LSH ran in about 110 s on 2,000 × 2,000, and its cost should grow sub-linearly on larger data.

## Running the notebook

The project runs in **Google Colab** with PySpark on a single local node. The speedups above are therefore limited to the cores of one machine and don't reflect a real cluster.

1. Open `spark_project.ipynb` in Colab or Jupyter.
2. Run the cells in order. The notebook:
   - installs `pyspark`
   - downloads and extracts SIFT1M from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`
   - builds the LSH join, runs the brute-force baseline, and reports recall and speedup
3. To change the experiment, edit the parameters cell:
   ```python
   L = 15      # number of hash tables
   k = 4       # hash functions per table
   w = 600     # bucket width
   theta = 300 # distance threshold
   ```

Each run writes a summary to `results_k{k}_L{L}_w{w}_{theta}.txt`.

**Requirements:** Python 3, `pyspark`, `numpy`, `matplotlib`. Spark driver and executor memory are set to 8 GB.

## Repository contents

| File | Description |
|------|-------------|
| `spark_project.ipynb` | Full implementation: LSH join, baseline, evaluation, plots |
| `spark report.pdf` | Project report (in Greek) |

## References

1. Indyk, P. and Motwani, R. 1998. *Approximate nearest neighbor: towards removing the curse of dimensionality.* STOC.
2. Datar, M., Immorlica, N., Indyk, P., and Mirrokni, V.S. 2004. *Locality-Sensitive Hashing Scheme Based on p-Stable Distributions.* SCG '04, 253–262. https://doi.org/10.1145/997817.997857
3. Zaharia, M. et al. 2016. *Apache Spark: A Unified Engine for Big Data Processing.* CACM 59(11), 56–65. https://doi.org/10.1145/2934664
4. Jégou, H., Douze, M., and Schmid, C. 2011. *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1), 117–128. https://doi.org/10.1109/TPAMI.2010.57
5. Rajaraman, A., Leskovec, J., and Ullman, J.D. 2014. *Mining of Massive Datasets.* Cambridge University Press. http://www.mmds.org
