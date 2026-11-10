"""4.1.3-4.1.4 - K-Means visual styles fitted on the development catalogue only."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)


def fit_kmeans(X_dev, k, seed, n_init=20, max_iter=500):
    """(6) on L2-normalised embeddings (Euclidean K-Means ~ spherical K-Means)."""
    return KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed).fit(X_dev)


def assign(X, centroids):
    """(11) nearest frozen centroid; also used for products released after the catalogue cutoff."""
    d = (X ** 2).sum(1)[:, None] - 2 * X @ centroids.T + (centroids ** 2).sum(1)[None, :]
    return d.argmin(1)


def stability(X_dev, k, seeds, n_init=20, max_iter=500):
    """4.1.4 - pairwise ARI (60) between runs that differ only in the random seed."""
    labels = [fit_kmeans(X_dev, k, s, n_init, max_iter).labels_ for s in seeds]
    return np.array([adjusted_rand_score(labels[i], labels[j])
                     for i in range(len(labels)) for j in range(i + 1, len(labels))])


def evaluate_grid(X_dev, k_grid, seed, stability_seeds, n_init=20, max_iter=500):
    """(8)-(10) + stability + size distribution for every candidate K."""
    rows = []
    for k in k_grid:
        km = fit_kmeans(X_dev, k, seed, n_init, max_iter)
        sizes = np.bincount(km.labels_, minlength=k)
        aris = stability(X_dev, k, [seed + i for i in range(stability_seeds)], n_init, max_iter)
        rows.append({"k": k,
                     "silhouette": silhouette_score(X_dev, km.labels_, metric="euclidean"),
                     "davies_bouldin": davies_bouldin_score(X_dev, km.labels_),
                     "inertia": km.inertia_,
                     "ari_mean": aris.mean(), "ari_min": aris.min(), "ari_sd": aris.std(),
                     "size_min": sizes.min(), "size_median": np.median(sizes), "size_max": sizes.max()})
    return pd.DataFrame(rows)


def attribute_agreement(labels, products, attributes=("category", "color", "fabric")):
    """How much the visual partition merely re-discovers catalogue attributes (4.8.2)."""
    return {a: {"nmi": normalized_mutual_info_score(products[a], labels),
                "ari": adjusted_rand_score(products[a], labels)} for a in attributes}


def passing_candidates(grid, coverage, rules):
    """Pre-registered gate (G2). The final K is still chosen by a human after the contact sheets."""
    g = grid.merge(coverage, on="k", how="left")
    g["pass_size"] = g["size_min"] >= rules["min_cluster_size"]
    g["pass_stability"] = g["ari_mean"] >= rules["min_stability_ari"]
    g["pass_coverage"] = g["min_eligible_dev_origins"] >= rules["min_eligible_dev_origins_per_style"]
    g["pass_all"] = g[["pass_size", "pass_stability", "pass_coverage"]].all(axis=1)
    return g


def profile(labels, products, attributes=("category", "color", "fabric", "season"), top=3):
    """Attribute composition of each cluster, used only to interpret and name styles (4.1.1)."""
    df = products.assign(style=labels)
    rows = []
    for k, g in df.groupby("style"):
        row = {"style": k, "n_products": len(g)}
        for a in attributes:
            shares = g[a].value_counts(normalize=True).head(top)
            row[f"top_{a}"] = "; ".join(f"{v} {s:.0%}" for v, s in shares.items())
        rows.append(row)
    return pd.DataFrame(rows)
