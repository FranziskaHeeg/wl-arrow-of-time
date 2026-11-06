"""Temporal graph kernels (Oettershagen et al., SDM 2020) on the class 1/3 vs 2/4 task.

Uses the authors' reference implementation, vendored in `tgkernel/` (https://gitlab.com/tgpublic/tgkernel):
temporal graphs are exported in TU format, the C++ tool computes the normalized Gram matrices of the
lifted kernels, and classification is done here with a C-SVM on the precomputed kernel.

By default, tgkernel's directed line graph expansion connects two events (u, v; t) and (v, w; t') whenever
t < t', i.e. it does not bound the waiting time (delta = infinity in the notation of the paper); this is the
configuration used for the results in the submitted paper. With `delta`, the expansion is restricted to
t < t' <= t + delta (see tgkernel/PATCHES.md), i.e. it becomes the temporal event graph of Definition 4.
"""

import json
import os
import pickle
import subprocess

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

DATA_ROOT = "data/processed/temporal"
TU_ROOT = "cache/tgkernel_data"
TGKERNEL_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tgkernel", "release", "tgkernel")
POSITIVE_CLASSES = (1, 3)
NEGATIVE_CLASSES = (2, 4)

# tgkernel action id and the gram-file infix it writes (see tgkernel/src/tgkernel_main.cpp).
# Only the directed line graph expansion (Sec. 3.2) is used here; the reference implementation of the
# other transformations crashes on this data (see export_tu_format).
KERNELS = {"DL-WL": (7, "__LGWL_")}
C_GRID = [10.0**e for e in range(-3, 4)]


def load_slice(data, slice, root=DATA_ROOT, limit=None):
    """Load the temporal graphs of a slice with labels 1 (classes 1/3) and -1 (classes 2/4)."""
    folder = os.path.join(root, data, f"slice{slice}")
    graphs = []
    for label, classes in ((1, POSITIVE_CLASSES), (-1, NEGATIVE_CLASSES)):
        for c in classes:
            files = sorted(f for f in os.listdir(os.path.join(folder, f"class{c}")) if f.endswith(".pickle"))
            for file in files[:limit]:
                with open(os.path.join(folder, f"class{c}", file), "rb") as f:
                    graphs.append((pickle.load(f), label))
    return graphs


def export_tu_format(graphs, prefix, undirected=False):
    """Write graphs in the TU format read by tgkernel's DataManager.

    One node line per node (`time,label`, both 0 since our graphs are unlabelled), one line per
    temporal edge in `_A.txt` with its time stamp in `_edge_attributes.txt`.

    Events are written as directed, matching how the EGNN and DBGNN pipelines treat them. The paper
    models temporal graphs as undirected and its reference implementation relies on that: the WL code
    dereferences the result of an edge-label lookup without checking it (`ColorRefinementKernel.cpp`),
    and the static expansion assumes the time vertex `(u, t+1)` exists. Those transformations
    therefore segfault on directed input; `undirected=True` writes every event in both directions,
    which is the model of Def. 2 and avoids the crashes. The directed line graph expansion used here
    works either way — with `undirected=True` it produces the two vertices per temporal edge of Def. 2.
    """
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    indicator, node_labels, edges, times, graph_labels = [], [], [], [], []

    offset = 0
    for graph_id, (g, label) in enumerate(graphs, start=1):
        node_idx = {node_id: offset + i + 1 for i, node_id in enumerate(g.nodes)}
        for _ in node_idx:
            indicator.append(str(graph_id))
            node_labels.append("0,0")

        # tgkernel parses time stamps with stoi, which overflows on the shifted times of classes 3/4.
        # Shifting each graph to start at 0 keeps all time differences — and hence its directed line
        # graph expansion — unchanged, since times are only ever compared within a graph.
        events = list(g.temporal_edges)
        t0 = min(int(t) for _, _, t in events)
        for u, v, t in events:
            edges.append(f"{node_idx[u]}, {node_idx[v]}")
            times.append(str(int(t) - t0))
            if undirected:
                edges.append(f"{node_idx[v]}, {node_idx[u]}")
                times.append(str(int(t) - t0))
        graph_labels.append(str(label))
        offset += len(node_idx)

    for suffix, lines in (
        ("_graph_indicator.txt", indicator),
        ("_node_labels.txt", node_labels),
        ("_A.txt", edges),
        ("_edge_attributes.txt", times),
        ("_graph_labels.txt", graph_labels),
    ):
        with open(prefix + suffix, "w") as f:
            f.write("\n".join(lines) + "\n")
    return prefix


def compute_grams(prefix, kernel="DL-WL", hmin=0, hmax=5, binary=TGKERNEL_BIN, delta=None):
    """Run the tgkernel binary and return the normalized Gram matrix files, one per WL iteration."""
    if not os.path.exists(binary):
        raise FileNotFoundError(f"tgkernel binary not found at {binary} — see tgkernel/PATCHES.md for build steps")
    action, infix = KERNELS[kernel]
    workdir = os.path.dirname(os.path.abspath(prefix))
    name = os.path.basename(prefix)

    subprocess.run(
        [binary, os.path.abspath(prefix), str(action), str(hmin), str(hmax)],
        cwd=workdir,
        check=True,
        stdout=subprocess.DEVNULL,
        env={**os.environ, **({"TGKERNEL_MAX_WAITING_TIME": str(delta)} if delta is not None else {})},
    )
    return [os.path.join(workdir, f"{name}{infix}{h}.gram") for h in range(hmin, hmax + 1)]


def load_gram(path):
    """Read a libsvm-style precomputed kernel file written by tgkernel.

    Raises ValueError on a file left truncated by a crashed run, so that callers recompute it instead
    of training on a partial kernel.
    """
    labels, rows = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            labels.append(int(parts[0]))
            rows.append([float(p.split(":")[1]) for p in parts[2:]])

    K = np.array(rows)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"{path} is not a square Gram matrix (shape {K.shape}) — likely truncated")
    return K, np.array(labels)


def get_grams(data, slice, kernel="DL-WL", hmin=0, hmax=5, root=DATA_ROOT, cache=True, limit=None, delta=None):
    """Export a slice (if needed) and return `(list of Gram matrices, labels)`, one matrix per h.

    `delta` bounds the waiting time of the line graph expansion (default: unbounded).
    """
    name = f"{data}_slice{slice}" + (f"_delta{delta}" if delta is not None else "")
    prefix = os.path.join(TU_ROOT, name)
    _, infix = KERNELS[kernel]
    files = [os.path.join(TU_ROOT, f"{name}{infix}{h}.gram") for h in range(hmin, hmax + 1)]

    if cache and all(os.path.exists(f) for f in files):
        try:
            grams, labels = zip(*(load_gram(f) for f in files))
            return list(grams), labels[0]
        except ValueError:  # truncated by a crashed run — recompute
            pass

    graphs = load_slice(data, slice, root=root, limit=limit)
    export_tu_format(graphs, prefix)
    files = compute_grams(prefix, kernel=kernel, hmin=hmin, hmax=hmax, delta=delta)
    grams, labels = zip(*(load_gram(f) for f in files))
    if len(labels[0]) != len(graphs):
        # tgkernel skips graphs whose line graph has no edges
        print(f"warning: {len(graphs) - len(labels[0])} of {len(graphs)} graphs of {name} have an empty line graph "
              f"and are not in the kernel", flush=True)
    return list(grams), labels[0]


def _select_params(grams, y, idx_train, folds=5, seed=None):
    """Pick the WL iteration h and the C parameter by cross validation on the training indices."""
    best = (-np.inf, 0, 1.0)
    for h, K in enumerate(grams):
        search = GridSearchCV(
            SVC(kernel="precomputed"),
            {"C": C_GRID},
            cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        )
        search.fit(K[np.ix_(idx_train, idx_train)], y[idx_train])
        if search.best_score_ > best[0]:
            best = (search.best_score_, h, search.best_params_["C"])
    return best[1], best[2]


def _fit_score(grams, y, idx_train, idx_test, h, C):
    svm = SVC(kernel="precomputed", C=C).fit(grams[h][np.ix_(idx_train, idx_train)], y[idx_train])
    return svm.score(grams[h][np.ix_(idx_test, idx_train)], y[idx_test])


def evaluate_holdout(grams, y, test_size=0.2, seed=None, folds=5):
    """Train/test split as in the EGNN/DBGNN experiments; h and C chosen by CV on the training part."""
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    h, C = _select_params(grams, y, idx_train, folds=folds, seed=seed)
    return _fit_score(grams, y, idx_train, idx_test, h, C), {"h": h, "C": C}


def evaluate_cv(grams, y, folds=10, repetitions=10, seed=0, inner_folds=10):
    """Protocol of the paper: repeated 10-fold CV, h and C selected fold-wise on the training folds."""
    accuracies = []
    for rep in range(repetitions):
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed + rep)
        fold_accuracies = []
        for idx_train, idx_test in cv.split(np.zeros(len(y)), y):
            h, C = _select_params(grams, y, idx_train, folds=inner_folds, seed=seed + rep)
            fold_accuracies.append(_fit_score(grams, y, idx_train, idx_test, h, C))
        accuracies.append(float(np.mean(fold_accuracies)))
    return float(np.mean(accuracies)), float(np.std(accuracies))


def run_experiment(data, slice, kernel="DL-WL", hmin=0, hmax=5, seed=None, verbose=True):
    """Compute the kernel for one slice and report SVM accuracy on a stratified 80/20 split."""
    grams, y = get_grams(data, slice, kernel=kernel, hmin=hmin, hmax=hmax)
    acc, params = evaluate_holdout(grams, y, seed=seed)
    if verbose:
        print(f"{data} slice{slice} {kernel}: acc {acc:.3f} (h={params['h']}, C={params['C']})", flush=True)
    return acc, params
