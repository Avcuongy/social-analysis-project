from pathlib import Path
import json
import logging
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_COMPLETED = DATA_DIR / "completed"
DATA_GITHUB_NETWORK = DATA_DIR / "github_social_network"
LOG_FILE = PROJECT_ROOT / "logs" / "data.log"
RANDOM_STATE = 42


def _load_edges(edges_path: Path) -> pd.DataFrame:
    df = pd.read_csv(edges_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def _prepare_processed_dataframe(
    target_path: Path,
    edges_path: Path,
    features_path: Path,
    processed_path: Path,
) -> pd.DataFrame:
    df_target = pd.read_csv(target_path)
    df_target.columns = df_target.columns.str.strip().str.lower().str.replace(" ", "_")

    df_edges = pd.read_csv(edges_path)
    df_edges.columns = df_edges.columns.str.strip().str.lower().str.replace(" ", "_")

    with open(features_path, "r", encoding="utf-8") as file_handle:
        features_json = json.load(file_handle)

    df_target = df_target.rename(columns={"id": "source"})
    df_edges = df_edges.rename(columns={"id_1": "source", "id_2": "target"})

    df = df_edges.merge(df_target, on="source", how="left")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)

    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Loaded %s feature entries from %s", len(features_json), features_path)

    return df


def _build_graph(df_edges: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df_edges, source="source", target="target", create_using=nx.Graph()
    )
    return G


def _sample_positive_edges(
    edges: List[Tuple[int, int]],
    test_ratio: float,
    graph: nx.Graph,
    random_state: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Set[Tuple[int, int]]]:
    rng_local = np.random.default_rng(random_state)

    n_test = int(len(edges) * test_ratio)

    bridge_edges = {tuple(sorted(edge)) for edge in nx.bridges(graph)}

    non_bridge_edges = [edge for edge in edges if edge not in bridge_edges]

    if len(non_bridge_edges) >= n_test:
        test_indices = rng_local.choice(
            len(non_bridge_edges), size=n_test, replace=False
        )
        test_pos = [non_bridge_edges[index] for index in test_indices]
    else:
        test_pos = list(non_bridge_edges)
        remaining = n_test - len(test_pos)
        bridge_pool = [edge for edge in edges if edge in bridge_edges]
        if remaining > len(bridge_pool):
            raise ValueError(
                "Not enough positive edges to create the requested test split."
            )
        bridge_indices = rng_local.choice(
            len(bridge_pool), size=remaining, replace=False
        )
        test_pos.extend([bridge_pool[index] for index in bridge_indices])

    test_pos = [tuple(edge) for edge in test_pos]
    test_set = set(test_pos)
    train_pos = [edge for edge in edges if edge not in test_set]

    return train_pos, test_pos, bridge_edges


def _sample_negative_pairs(
    graph: nx.Graph,
    n_samples: int,
    forbidden_pairs: Set[Tuple[int, int]],
    random_state: int,
) -> List[Tuple[int, int]]:
    rng_local = np.random.default_rng(random_state)
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least two nodes.")

    sampled_pairs: List[Tuple[int, int]] = []
    sampled_set: Set[Tuple[int, int]] = set()
    max_attempts = max(n_samples * 20, 1000)
    attempts = 0

    while len(sampled_pairs) < n_samples and attempts < max_attempts:
        left_index, right_index = rng_local.choice(len(nodes), size=2, replace=False)
        u = nodes[left_index]
        v = nodes[right_index]
        pair = tuple(sorted((u, v)))

        if pair in forbidden_pairs or pair in sampled_set or graph.has_edge(u, v):
            attempts += 1
            continue

        sampled_set.add(pair)
        sampled_pairs.append(pair)
        attempts += 1

    if len(sampled_pairs) < n_samples:
        raise ValueError(
            "Not enough negative pairs available for the requested sample size."
        )

    return sampled_pairs


def _validate_disjoint_pairs(
    negative_pairs: List[Tuple[int, int]],
    positive_pairs: Set[Tuple[int, int]],
    split_name: str,
) -> None:
    overlap = set(negative_pairs).intersection(positive_pairs)
    if overlap:
        raise ValueError(
            f"{split_name} negatives overlap with positive edges: {len(overlap)} pairs."
        )


def _create_link_prediction_dataset(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:

    df = df.drop(columns=["name"], errors="ignore")

    G = _build_graph(df)

    all_edges = [tuple(sorted((u, v))) for u, v in G.edges()]
    all_edges = list(dict.fromkeys(all_edges))

    train_pos, test_pos, bridge_edges = _sample_positive_edges(
        edges=all_edges, test_ratio=test_ratio, graph=G, random_state=random_state
    )

    positive_forbidden = set(all_edges)

    train_neg = _sample_negative_pairs(
        graph=G,
        n_samples=len(train_pos),
        forbidden_pairs=positive_forbidden,
        random_state=random_state,
    )

    _validate_disjoint_pairs(train_neg, positive_forbidden, "train")

    negative_forbidden = positive_forbidden.union(train_neg)

    test_neg = _sample_negative_pairs(
        graph=G,
        n_samples=len(test_pos),
        forbidden_pairs=negative_forbidden,
        random_state=random_state,
    )

    _validate_disjoint_pairs(test_neg, positive_forbidden, "test")

    train_df = pd.DataFrame(
        {
            "source": [u for u, _ in train_pos + train_neg],
            "target": [v for _, v in train_pos + train_neg],
            "label": [1] * len(train_pos) + [0] * len(train_neg),
            "split": ["train"] * (len(train_pos) + len(train_neg)),
            "type": (["positive"] * len(train_pos) + ["negative"] * len(train_neg)),
        }
    )

    test_df = pd.DataFrame(
        {
            "source": [u for u, _ in test_pos + test_neg],
            "target": [v for _, v in test_pos + test_neg],
            "label": [1] * len(test_pos) + [0] * len(test_neg),
            "split": ["test"] * (len(test_pos) + len(test_neg)),
            "type": (["positive"] * len(test_pos) + ["negative"] * len(test_neg)),
        }
    )

    link_prediction_df = pd.concat([train_df, test_df], ignore_index=True)

    return link_prediction_df


def strategy(
    processed_rel_path: str = "data/processed/processed.csv",
    target_rel_path: str = "data/github_social_network/musae_git_target.csv",
    edges_rel_path: str = "data/github_social_network/musae_git_edges.csv",
    features_rel_path: str = "data/github_social_network/musae_git_features.json",
    train_out_rel_path: str = "data/completed/train.csv",
    test_out_rel_path: str = "data/completed/test.csv",
    test_ratio: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    LOGGER = logging.getLogger(__name__)

    LOGGER.info("Starting link prediction dataset creation")
    print("[Strategy] Starting link prediction dataset creation")

    project_root = Path(__file__).resolve().parents[2]
    processed_path = project_root / processed_rel_path
    target_path = project_root / target_rel_path
    edges_path = project_root / edges_rel_path
    features_path = project_root / features_rel_path
    train_out_path = project_root / train_out_rel_path
    test_out_path = project_root / test_out_rel_path

    LOGGER.info("Preparing processed data from raw inputs")
    df = _prepare_processed_dataframe(
        target_path=target_path,
        edges_path=edges_path,
        features_path=features_path,
        processed_path=processed_path,
    )

    LOGGER.info("Creating link prediction dataset")
    link_prediction_df = _create_link_prediction_dataset(
        df=df, test_ratio=test_ratio, random_state=random_state
    )

    train_df = link_prediction_df[link_prediction_df["split"] == "train"].copy()
    test_df = link_prediction_df[link_prediction_df["split"] == "test"].copy()

    df_indexed = df.set_index(["source", "target"])
    train_df = train_df.join(df_indexed, on=["source", "target"], how="left")
    test_df = test_df.join(df_indexed, on=["source", "target"], how="left")

    train_df = train_df.drop(columns=["split", "type"], errors="ignore")
    test_df = test_df.drop(columns=["split", "type"], errors="ignore")

    train_out_path.parent.mkdir(parents=True, exist_ok=True)
    test_out_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out_path, index=False)
    test_df.to_csv(test_out_path, index=False)

    print(f"[Strategy] Completed train dataset to {train_out_path}")
    print(f"[Strategy] Completed test dataset to {test_out_path}")
    LOGGER.info("Saved train dataset to %s", train_out_path)
    LOGGER.info("Saved test dataset to %s", test_out_path)


if __name__ == "__main__":
    strategy()
