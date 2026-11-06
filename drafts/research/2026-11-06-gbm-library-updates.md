---
title: "State of gradient boosting libraries — November 2026"
date: 2026-11-06
status: draft
type: research
tags: [gradient-boosting, xgboost, lightgbm, catboost, scikit-learn, tabular-ml]
---

# State of gradient boosting libraries — November 2026

*Research brief, compiled 6 November 2026. Versions and dates are from PyPI and GitHub release pages checked that day. PyPI/GitHub timestamps are UTC, so dates are given as UTC calendar days.*

## Executive summary

All four mainstream GBDT stacks shipped meaningful releases in the past 12 months, and the gaps between them have narrowed. **XGBoost** moved fastest. The 3.x line (3.1 → 3.4.1) turned categorical support from experimental to on-by-default, added a model-embedded category re-coder, made GPU external-memory training practical, and completed its **vector-leaf (multi-output tree)** model. **LightGBM 4.7.0** (July 2026) was its first feature release in 17 months. It brought native polars/Arrow input, multi-GPU training via NCCL, first ROCm builds, and a move from `Microsoft/LightGBM` to `lightgbm-org/LightGBM`. **CatBoost 1.2.9/1.2.10** (February 2026) was a maintenance-plus release: polars input, Python 3.14, scikit-learn 1.8 compatibility, and Spark 4.x support. **scikit-learn** has had no major HistGradientBoosting features since 1.7 (validation-set early stopping). 1.9 fixed weighted bin-edge computation and deprecated `criterion` in the classic `GradientBoosting*`. On the official TabArena leaderboard, tuned-and-ensembled LightGBM and CatBoost are the top tree models, roughly level with each other, with XGBoost slightly behind. All three now rank well below the newest tabular foundation models on that small-to-medium IID benchmark. The BeyondArena study, though, finds tree models still dominate on non-IID, large, and high-dimensional data, and that is where most production systems run. The data doesn't show a single winner. The right choice depends on data shape, hardware, and deployment constraints.

---

## 1. XGBoost

**Latest stable:** 3.4.1 on PyPI (uploaded 2026-08-15; release notes date it Aug 14, 2026). GitHub also has **v3.4.2 and v3.3.1 (2026-09-15)**. Per their notes they are "identical to 3.4.1/3.3.0 … except that [they] now support Python 3.11". As of today, PyPI still lists 3.4.1 as latest, and its `requires_python` is `>=3.12`. *I could not confirm which channel distributes 3.4.2.*

Release cadence over the past 12 months: 3.1.0 (Sep 22, 2025) → 3.1.1–3.1.3 → **3.2.0 (Feb 9, 2026)** → **3.3.0 (Jun 17, 2026)** → **3.4.0 (Aug 4, 2026)** → 3.4.1.

### What changed

| Release | Highlights |
|---|---|
| **3.1.0** (Sep 2025) | **Categorical re-coder**: categories (including strings) are stored in the model and data is re-coded at inference for consistent encoding. Categorical support for **Polars**. The *experimental* tag was removed from categorical support. Adaptive CPU/GPU cache for **CUDA external memory**. Optional nvcomp / GB200 decompression. Vector intercept (`base_score`) for multi-output models. **Breaking:** the binary model format was removed, and the old GPU params (`gpu_id`, `gpu_hist`, `use_gpu`…) are gone; use `device`. New `xgboost-cu13` PyPI package. Faster CPU inference via array-based tree traversal. |
| 3.1.1–3.1.3 | SHAP group-ID fix for vector intercepts. `enable_categorical` inferred on model load. scikit-learn 1.8 compatibility. **ARM (aarch64) CUDA wheels on PyPI**. |
| **3.2.0** (Feb 2026) | Major vector-leaf progress: reduced-gradient ("sketch boost") splits, all regression objectives, GPU parity, sampling, external memory. CUDA async memory pool for external memory (no RMM required, Linux only). **Deprecated CLI removed**. Adaptive CPU histogram block size, "up to 2x speedup for certain workloads". Pandas 3.0 support. CUDA 12.9 build. |
| **3.3.0** (Jun 2026) | **Categorical support enabled by default** (`enable_categorical` still available to disable it). New **QuadratureTreeSHAP** implementation for "improved numerical stability and faster execution", plus exact SHAP for vector-leaf models on CPU/GPU. New **expectile regression** objective (`reg:expectileerror`). Refactored quantile sketch with lower GPU memory use. Distributed training no longer needs all-to-all connections. Reproducible RNG state is serialized. **`gblinear` deprecated**. **Minimum Python 3.12**. PySpark support for Spark Connect ML. |
| **3.4.0** (Aug 2026) | **Vector-leaf feature-complete** for `hist` (now "experimental" rather than "work in progress"): categoricals, DART, monotone/interaction constraints, distributed/Dask. Quantile and MAE objectives now use smooth approximations instead of line search for leaf values. **Breaking:** column-split support removed, federated plugin dropped from wheels, RF wrappers deprecated (use `num_parallel_tree`). **Default PyPI wheels are built with CUDA 13.3**; use `xgboost-cu12` for CUDA 12.9. Windows on ARM64 wheels. |

### Why it matters

- **Categorical handling is now first-class and safer in production.** The re-coder fixes a long-standing failure mode where category codes differ between training and serving. Native categoricals (on by default since 3.3) remove much one-hot or label-encoding boilerplate.
- **Multi-output and multiclass models get a real alternative.** Per the maintainers' Aug 25, 2026 blog post, `multi_strategy="multi_output_tree"` achieved lower held-out loss on 9 of 10 multiclass datasets and used fewer splits and less disk space on all 10. They now call it "a compelling default to try" for multiclass. It may need more boosting rounds and a lower learning rate.
- **Upgrade friction is real.** The 3.x line removed old binary models, legacy GPU params, the CLI, and column-split. It also raised the Python floor to 3.12 and switched the default CUDA to 13.x. Pin versions and re-save models as JSON/UBJSON before upgrading old pipelines.
- **GPU / large-data story:** external-memory training on GPU (including Grace-Hopper/GB200-class C2C systems) is a sustained focus. That makes XGBoost the most developed option for data that doesn't fit in GPU memory.

### Practitioner signal
- XGBoost maintainers' vector-leaf post (xgboost.ai, Aug 2026) and a Numerai forum thread experimenting with vector-leaf two-target models (seen in search results; not read in full).
- The mlcontests.com 2025 review (r/MachineLearning, early 2026) reports that GBDTs, "in particular, XGBoost and LightGBM, and to a slightly lesser extent, Catboost", were "still the go-to for most tabular problems". AutoGluon and TabPFN appeared in some winning solutions.

---

## 2. LightGBM

**Latest stable:** **4.7.0** (GitHub release 2026-07-18; PyPI upload 2026-07-18). Previous: 4.6.0 (2025-02-15). No pre-release found.

### What changed

**4.6.0 (Feb 2025):** NVIDIA Blackwell support, Python 3.13, linear trees on `device=gpu`, **bagging by query for LambdaRank**, and fewer copies of column-major NumPy arrays. Breaking: `feature_name` / `categorical_feature` can no longer be passed via `train()` / `cv()`, and scikit-learn ≥0.24.2 is required.

**4.7.0 (Jul 2026), highlights from the release notes:**
- **Polars and Arrow input** in the Python package via `narwhals` and the Arrow PyCapsule interface. PyArrow Tables are accepted in the scikit-learn API.
- **Significantly improved GPU support:** **multi-GPU training for NVIDIA via NCCL** (#6138), the **first ROCm/HIP builds** for AMD GPUs (#6086), CUDA 13 support, and a CUDA binary-error metric.
- **Dask / distributed:** "lots of bugfixes … still ongoing work".
- scikit-learn API: new `eval_X` / `eval_y` in `fit()`, **`eval_set` deprecated**. `LGBMClassifier.decision_function()` added.
- **Breaking:** minimum Python 3.10 (3.7/3.8 dropped), C++17 default, H2O datatable support dropped, R 3.x dropped, lazy imports for dask/graphviz/matplotlib etc.
- Fixes: weighted percentile calculation for `mape` / `quantile` / `regression_l1` objectives. `regression_l1` is now prohibited with linear trees.
- **Organizational:** the repo moved **`Microsoft/LightGBM` → `lightgbm-org/LightGBM`** in March 2026, and the default branch changed `master` → `main`. Per maintainer issue #7187, Microsoft's Open Source team "were considering archiving `Microsoft/LightGBM`". The maintainers (including the creator) moved it. License (MIT), PyPI/CRAN/NuGet names, and the Java namespace are unchanged. The maintainers now pay for some GPU CI themselves.

### Why it matters
- The 17-month gap between 4.6 and 4.7 and the org move raised governance questions. The outcome looks healthy: same maintainers, active release, and a closed migration issue with "no reports of this causing problems".
- **GPU is no longer LightGBM's weak spot on paper.** Multi-GPU CUDA and ROCm are new. Independent 2026 benchmarks of the new multi-GPU path were not found. Treat performance claims as unverified until you test on your own hardware.
- The `eval_set` deprecation will touch most existing scikit-learn-API training scripts. Expect warnings now and breakage later.
- Upgrade note: anything still pointing at `github.com/microsoft/LightGBM` (CI scripts, submodules, docs) should be updated, though GitHub redirects transferred repos.

---

## 3. CatBoost

**Latest stable:** **1.2.10** (PyPI 2026-02-18; GitHub release 2026-02-19), which followed **1.2.9** (2026-02-17/18). Before that: 1.2.8 (2025-04-13). There has been no release since February 2026, but the repository shows commits as recently as 2026-11-06. No 1.3 pre-release was found.

### What changed
**1.2.9 (Feb 2026):**
- **Polars input support** for features, labels, and auxiliary data (weights, timestamps).
- **Python 3.14**. Estimators gained `__sklearn_tags__` for **scikit-learn ≥1.8** compatibility, plus several `get_params` / tag fixes.
- New `RMSPE` loss/metric (CPU-only). C/C++ applier `LoadFullModelZeroCopy` (mmap) for low-memory model loading.
- **Speedups:** `Lossguide` grow policy on CPU about 1.4x faster. 128-thread data-loading limit removed. Windows builds now enable the SSE flag properly, which affected some training and inference paths.
- Rust package: Windows support, `Sync` model.

**1.2.10 (Feb 2026):** **Spark 4.0.x / 4.1.x** support and JVM `predictTransposed`. JVM artifacts missing from 1.2.9 were published here.

**1.2.8 (Apr 2025)** is background: GPU custom metrics fixes, Spark 2.x removed, and the ARM probability-NaN bug fix also noted in the Node package.

### Why it matters
- CatBoost's core value hasn't changed: **ordered target statistics for categoricals and strong defaults**. 2026 work is about ecosystem compatibility (polars, sklearn 1.8, Spark 4, Python 3.14), not new modeling features.
- On TabArena, **CatBoost (default)** scores Elo 1358, above *tuned* XGBoost (1333) and close to tuned LightGBM (1365). It remains the strongest "no tuning" GBDT on that benchmark.
- An MDPI *Algorithms* 2026 study of label-noise robustness (15 OpenML-CC18 datasets) found CatBoost degraded significantly less than LightGBM at 40% symmetric and asymmetric label noise. No ranking held under instance-dependent noise.
- Trade-off: training time. On TabArena, median train time for tuned CatBoost is about 1346 s per 1K rows of the benchmark protocol, versus about 417 s for LightGBM and about 693 s for XGBoost. These are benchmark-protocol numbers, not single-fit times.

---

## 4. scikit-learn HistGradientBoosting (and related)

**Latest stable:** **scikit-learn 1.9.1** (PyPI 2026-09-10). 1.9.0 shipped 2026-06-02 (rc1 2026-05-20). 1.8.0 shipped 2025-12-10.

### GBM-relevant changes
- **1.7 (Jun 2025):** `HistGradientBoosting*` `fit()` accepts **`X_val`, `y_val`, `sample_weight_val`** for explicit early-stopping validation data (#27124). This was the most useful HGB API addition of the period.
- **1.8 (Dec 2025):** no HGB-specific entries in the changelog. Release headlines were **Array API support (GPU computation for some estimators)**, **free-threaded CPython 3.14 support**, and temperature scaling in `CalibratedClassifierCV`. I did not find evidence that HGB itself gained Array API/GPU support.
- **1.9 (Jun 2026):**
  - **Fix:** HGB bin edges now handle `sample_weight` properly and consistently, using weight-aware quantiles (averaged inverted CDF), or weighted subsampling when `n_samples > subsample` (#29641). *If you train weighted HGB models, retrain and compare after upgrading, because splits can change.*
  - **Fix + API change:** classic `GradientBoostingRegressor/Classifier` with the default `criterion="friedman_mse"` was "computing impurity values with an incorrect scaling, leading to unexpected trees in some cases". It now uses `squared_error`, and **the `criterion` parameter is deprecated** (#32708). NGBoost 0.5.11 already followed this change.
  - Experimental **callbacks** API (`set_callbacks`, `ProgressBar`, `ScoringMonitor`). Only a few estimators support it so far. *I did not verify whether HGB is among them.*
- Earlier background: 1.6 parallelized bin-threshold search and added finer `verbose` control.

### Why it matters
HGB remains the default when you want **zero extra dependencies, full sklearn pipeline/metadata-routing integration, native missing-value and categorical handling, and monotonic constraints**. It is not moving toward GPU or distributed training. For those, use XGBoost or LightGBM. The 1.9 bin-edge fix is a quiet but real model-change risk for weighted workloads.

---

## 5. Cross-library comparison

### Release snapshot (as of 2026-11-06)

| Library | Latest stable (date, UTC) | Previous feature release | Python floor | Notable 2025–26 themes |
|---|---|---|---|---|
| XGBoost | 3.4.1 (2026-08-15, PyPI); 3.4.2 GitHub tag 2026-09-15 | 3.3.0 (2026-06-17) | ≥3.12 on PyPI 3.4.1 | Vector leaf, categorical by default, GPU external memory, CUDA 13 |
| LightGBM | 4.7.0 (2026-07-18) | 4.6.0 (2025-02-15) | ≥3.10 | Polars/Arrow, multi-GPU NCCL, ROCm, org move |
| CatBoost | 1.2.10 (2026-02-18) | 1.2.9 (2026-02-17) | Not checked | Polars, sklearn 1.8 tags, Spark 4, Py 3.14 |
| scikit-learn | 1.9.1 (2026-09-10) | 1.9.0 (2026-06-02) | Not checked | Weighted bin fix, callbacks, Array API (non-HGB) |

### TabArena leaderboard (official CSV, fetched 2026-11-06)

TabArena-v0.1 is a living benchmark of 51 curated IID datasets (AutoGluon team, NeurIPS 2025). Elo is anchored at RandomForest (default) = 1000. Selected tree-model rows:

| Model | Elo | Median train s/1K | Median predict s/1K |
|---|---|---|---|
| LightGBM (tuned + ensembled) | 1407 | 417 | 2.64 |
| CatBoost (tuned + ensembled) | 1397 | 1346 | 0.34 |
| CatBoost (tuned) | 1386 | 1346 | 0.038 |
| LightGBM (tuned) | 1365 | 417 | 0.33 |
| CatBoost (default) | 1358 | 5.9 | 0.025 |
| XGBoost (tuned + ensembled) | 1356 | 693 | 1.69 |
| XGBoost (tuned) | 1333 | 693 | 0.31 |
| EBM (tuned + ensembled) | 1252 | 1711 | 0.14 |
| XGBoost (default) | 1207 | 1.9 | 0.12 |
| LightGBM (default) | 1182 | 2.0 | 0.14 |
| PerpetualBooster (tuned + ensembled) | 1092 | 185 | 0.60 |

For context, the top entries on the same board are tabular foundation models: Kumo-Tabular 1938, LimiX-2 1928, TabPFN-3.5 1842 (GPU; several have non-commercial licenses). The best tree model ranks around 30th. Elo confidence intervals for the GBDTs are roughly ±25–38, so **LightGBM vs CatBoost at the top is a statistical tie**.

Caveat: BeyondArena (arXiv 2606.30410, Jun 2026; 142 datasets including temporal and grouped tasks) concludes that foundation models "excel on tiny- to medium-sized IID data, while traditional tree-based and deep learning models still dominate on non-IID, large, and high-dimensional datasets."

### Which library for which job

| Use case | Current best fit | Why (evidence) |
|---|---|---|
| Large tabular data, out-of-core / GPU memory limits | **XGBoost** | Sustained GPU external-memory work (3.1–3.4), adaptive host/GPU cache, async memory pool. |
| Fast iteration on CPU, wide or sparse data | **LightGBM** | Fastest tuned training among the big three on TabArena. Leads the tuned + ensembled GBDT Elo. |
| Heavy or high-cardinality categoricals, minimal tuning | **CatBoost** | Best default-config GBDT on TabArena. Ordered target statistics. More robust to label noise in the 2026 MDPI study. Slow to tune. |
| Multi-GPU training | **XGBoost** (mature: Dask/Spark/NCCL); **LightGBM 4.7** (new NCCL multi-GPU) | LightGBM's path is new. Benchmark it before committing. |
| AMD GPUs | **LightGBM 4.7** (first ROCm builds) | XGBoost ROCm status not verified in this pass. |
| Multiclass / multi-output | **XGBoost vector leaf** (try it), **Py-Boost** (GPU multi-output research) | Maintainer benchmarks: better loss on 9/10 multiclass sets. Still labeled experimental. |
| scikit-learn pipelines, no extra deps | **HistGradientBoosting** | Native NaN and categoricals, metadata routing, `X_val` early stopping (1.7+). All three big libraries also expose sklearn estimators and updated for sklearn 1.8 tags. |
| Polars-first data stacks | All three big libraries now (XGBoost 3.1+, CatBoost 1.2.9+, LightGBM 4.7+) | Polars/Arrow support arrived in all of them within about 10 months. |
| Learning-to-rank | **LightGBM** / **XGBoost** (LambdaMART); CatBoost YetiRank | LightGBM 4.6 added query-level bagging. XGBoost 3.4 fixed LTR pair sampling on continued training. No new head-to-head ranking benchmark found. |
| Interpretability | **XGBoost** (new QuadratureTreeSHAP; SHAP for vector leaf) or **EBM** (glass-box) | EBM ranks lower on TabArena (~1252) but is inherently interpretable. |
| Probabilistic / distributional output | **NGBoost**, or XGBoost quantile/expectile | XGBoost 3.3 added expectile regression. NGBoost 0.5.9 added a SymPy distribution factory. |
| High-throughput inference | Any, served via **NVIDIA nvForest** | Standalone successor to cuML FIL (announced 2026-09-03). Loads XGBoost/LightGBM/sklearn models on GPU or CPU. |

---

## 6. Emerging and adjacent libraries

Ranked by evidence of real use. The newer entrants have very little adoption so far.

- **NVIDIA nvForest** (inference only). Announced 2026-09-03 as the standalone successor to cuML's Forest Inference Library. It is about 50 MiB versus 250 MiB for cuML and has a CPU fallback with `device="auto"`. It loads XGBoost JSON/UBJ, LightGBM text, and sklearn forests via Treelite. LightGBM 4.7 docs now list it as an external repo. NVIDIA claims "2-3 orders of magnitude" faster than scikit-learn inference (vendor claim, unverified).
- **NGBoost** (Stanford ML Group, about 1.9k GitHub stars). Actively maintained: 0.5.8 → **0.5.11 (2026-06-26)**. Changes include a SymPy distribution factory (Beta, Beta-Binomial, Logit-Normal…), early stopping for `NGBClassifier` / `NGBSurvival`, and a per-parameter base learner. It remains the go-to for distributional and uncertainty outputs within sklearn.
- **PerpetualBooster** (Rust core, about 720 stars). It aims to eliminate hyperparameter tuning with a single `budget` parameter. 2.2.0 (2026-10-04) added SciPy sparse input and cgroup-aware memory limits. **2.3.0 (2026-10-24)** added Python 3.15. A 3.0.0 line has been in RC since March 2026. **It is on TabArena but ranks low** (Elo 934 default, 1092 tuned + ensembled), so the "no tuning needed" claim doesn't hold up against the big three on that benchmark.
- **Py-Boost** (Sber AI Lab, about 210 stars). GPU GBDT in Python/CuPy specialized for multi-output via SketchBoost. XGBoost 3.2's "reduced gradient (sketch boost)" vector-leaf feature uses the same idea. **0.6.0.dev1 (2026-11-02)** migrated CPU inference to Treelite 4 / TL2cgen. Last stable was 0.5.2 (2025-06-16).
- **InterpretML EBM** (interpret 0.7.8, 2026-03-17). A boosted GAM, the interpretable option. It is on TabArena (tree-based category).
- **GPBoost** 1.8.0 (combines boosting with Gaussian process / mixed effects). Only version metadata checked; no practitioner evidence gathered.
- **New 2026 GBDT entrants on TabArena:** **ChimeraBoost** (CatBoost-inspired, pure Python + numba; v0.34.0; 13 stars; Elo 1362 tuned + ensembled, roughly level with tuned+ensembled XGBoost), **CTBoost** (conditional-inference trees, C++/CUDA; 4 stars), **PrismBoost** (oblique SEFR splits; 1 star). They are benchmark-visible but have almost no adoption. Worth watching, not adopting.
- **ThunderGBM:** last PyPI release 0.3.17 (Sep 2022). **Inactive.** Don't start new work on it.
- **Snap ML** (IBM): 1.17.2 (Apr 2026), still released. No 2025–26 practitioner discussion found.
- **Tabular foundation models** (TabPFN 9.2.0 package, TabPFN-3.5, LimiX-2, Kumo-Tabular…) are not GBMs but are now the main competitive pressure. Practitioner consensus (r/MachineLearning threads, mlcontests 2025 report) is to treat them as a strong option on small-to-medium IID data and validate against a tuned GBDT. Several top models are GPU-only and non-commercially licensed.

---

## 7. What to do with this

1. **Upgrading XGBoost to 3.3+:** check for implicit categorical behavior changes (now on by default), legacy binary models, `gpu_hist`-style params, and Python < 3.12 environments.
2. **On LightGBM:** migrate `eval_set` → `eval_X` / `eval_y`, and update repo URLs to `lightgbm-org`.
3. **On weighted sklearn HGB:** re-validate after 1.9, because bin edges changed.
4. **For multiclass with many classes:** benchmark XGBoost `multi_strategy="multi_output_tree"` against the default.
5. **For small IID tables:** benchmark a tabular foundation model against a tuned GBDT, and check licensing before production use.

---

## Unverified or uncertain
- Where XGBoost **3.4.2** is distributed (GitHub tag only; not on PyPI as of today).
- Real-world performance of LightGBM 4.7 multi-GPU (NCCL) and ROCm builds. No independent benchmarks found.
- Whether scikit-learn 1.9 callbacks support HistGradientBoosting.
- XGBoost ROCm/AMD GPU status in 3.x (not checked).
- TabArena results are a moving target (living leaderboard, updated about daily). Numbers above are from the CSV fetched 2026-11-06.
- Third-party blog benchmarks (algorithmine.com, tildalice.io) appeared in search but were not used for claims, because their methodology couldn't be verified.

## Sources

**Primary: releases, changelogs, package indexes**
- XGBoost release notes: https://xgboost.readthedocs.io/en/latest/changes/v3.1.0.html · https://xgboost.readthedocs.io/en/latest/changes/v3.2.0.html · https://xgboost.readthedocs.io/en/latest/changes/v3.3.0.html · https://xgboost.readthedocs.io/en/latest/changes/v3.4.0.html
- XGBoost GitHub releases (3.4.2 / 3.3.1 notes): https://github.com/dmlc/xgboost/releases
- XGBoost vector-leaf blog (Aug 25, 2026): https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model
- XGBoost multi-output tutorial: https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html
- PyPI: https://pypi.org/project/xgboost/ · https://pypi.org/project/lightgbm/ · https://pypi.org/project/catboost/ · https://pypi.org/project/scikit-learn/
- LightGBM v4.7.0: https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0
- LightGBM releases (4.6.0): https://github.com/lightgbm-org/LightGBM/releases
- LightGBM org move: https://github.com/lightgbm-org/LightGBM/issues/7187
- CatBoost releases (1.2.9, 1.2.10, 1.2.8): https://github.com/catboost/catboost/releases
- scikit-learn changelogs: https://scikit-learn.org/stable/whats_new/v1.9.html · https://scikit-learn.org/stable/whats_new/v1.8.html · https://scikit-learn.org/stable/whats_new/v1.7.html · https://scikit-learn.org/stable/whats_new/v1.6.html
- scikit-learn release highlights: https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_9_0.html · https://scikit-learn.org/stable/auto_examples/release_highlights/plot_release_highlights_1_8_0.html
- NGBoost releases: https://github.com/stanfordmlgroup/ngboost/releases
- Perpetual releases: https://github.com/perpetual-ml/perpetual/releases
- Py-Boost: https://github.com/sb-ai-lab/Py-Boost
- nvForest: https://github.com/rapidsai/nvforest · https://docs.nvidia.com/nvforest/latest/blog/2026-09-03-supercharge-tree-inference/
- Other PyPI pages checked: https://pypi.org/project/ngboost/ · https://pypi.org/project/perpetual/ · https://pypi.org/project/py-boost/ · https://pypi.org/project/thundergbm/ · https://pypi.org/project/snapml/ · https://pypi.org/project/interpret/ · https://pypi.org/project/gpboost/ · https://pypi.org/project/tabpfn/ · https://pypi.org/project/chimeraboost/ · https://pypi.org/project/ctboost/ · https://pypi.org/project/prismboost/

**Benchmarks and papers**
- TabArena leaderboard (official): https://huggingface.co/spaces/TabArena/leaderboard (CSV: https://huggingface.co/spaces/TabArena/leaderboard/resolve/main/data/entrants_models/imputation_yes/splits_all/tasks_all/datasets_all/website_leaderboard.csv)
- TabArena paper: https://arxiv.org/abs/2506.16791 · code: https://github.com/autogluon/tabarena
- BeyondArena paper: https://arxiv.org/abs/2606.30410
- Label-noise robustness of XGBoost/LightGBM/CatBoost (MDPI *Algorithms*, 2026): https://doi.org/10.3390/a19090730
- New GBDT entrants: https://github.com/bbstats/chimeraboost · https://github.com/captnmarkus/ctboost · https://github.com/PrismBoost/PrismBoost

**Practitioner discussion**
- mlcontests 2025 competition analysis (r/MachineLearning): https://www.reddit.com/r/MachineLearning/comments/1r8y1ha/r_analysis_of_350_ml_competitions_in_2025/
- Michael Clark, "Is Boosting Still All You Need for Tabular Data?" (Mar 2026): https://m-clark.github.io/posts/2026-03-01-dl-for-tabular-foundational/
