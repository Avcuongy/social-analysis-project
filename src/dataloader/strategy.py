from pathlib import Path
import logging
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
LOG_FILE = PROJECT_ROOT / "logs" / "data.log"
RANDOM_STATE = 42


def _load_edges(edges_path: Path) -> pd.DataFrame:
    df = pd.read_csv(edges_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def _build_graph(df_edges: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_edgelist(
        df_edges, source="id_1", target="id_2", create_using=nx.Graph()
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


def _sample_hard_negative_pairs(
    graph: nx.Graph,
    n_samples: int,
    forbidden_pairs: Set[Tuple[int, int]],
    random_state: int,
) -> List[Tuple[int, int]]:
    rng_local = np.random.default_rng(random_state)
    nodes = list(graph.nodes())
    rng_local.shuffle(nodes)

    sampled_pairs: List[Tuple[int, int]] = []
    sampled_set: Set[Tuple[int, int]] = set()

    for mid_node in nodes:
        neighbors = list(graph.neighbors(mid_node))
        if len(neighbors) < 2:
            continue
        rng_local.shuffle(neighbors)
        for left_index in range(len(neighbors) - 1):
            for right_index in range(left_index + 1, len(neighbors)):
                u = neighbors[left_index]
                v = neighbors[right_index]
                pair = tuple(sorted((u, v)))
                if (
                    pair in forbidden_pairs
                    or pair in sampled_set
                    or graph.has_edge(u, v)
                ):
                    continue
                sampled_set.add(pair)
                sampled_pairs.append(pair)
                if len(sampled_pairs) == n_samples:
                    return sampled_pairs

    raise ValueError("Not enough hard-negative candidates with a shared neighbor.")


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
    df_edges: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:

    G = _build_graph(df_edges)

    all_edges = [tuple(sorted((u, v))) for u, v in G.edges()]
    all_edges = list(dict.fromkeys(all_edges))

    train_pos, test_pos, bridge_edges = _sample_positive_edges(
        edges=all_edges, test_ratio=test_ratio, graph=G, random_state=random_state
    )

    positive_forbidden = set(all_edges)

    train_neg = _sample_hard_negative_pairs(
        graph=G,
        n_samples=len(train_pos),
        forbidden_pairs=positive_forbidden,
        random_state=random_state,
    )

    _validate_disjoint_pairs(train_neg, positive_forbidden, "train")

    negative_forbidden = positive_forbidden.union(train_neg)

    test_neg = _sample_hard_negative_pairs(
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
    edges_rel_path: str = "data/github_social_network/musae_git_edges.csv",
    target_rel_path: str = "data/github_social_network/musae_git_target.csv",
    out_rel_path: str = "data/processed/processed.csv",
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
    edges_path = project_root / edges_rel_path
    out_path = project_root / out_rel_path

    LOGGER.info("Loading edges from %s", edges_path)
    df_edges = _load_edges(edges_path)

    LOGGER.info("Creating link prediction dataset")
    link_prediction_df = _create_link_prediction_dataset(
        df_edges=df_edges, test_ratio=test_ratio, random_state=random_state
    )

    target_path = project_root / target_rel_path

    if target_path.exists():
        LOGGER.info("Loading node features from %s", target_path)
        df_target = pd.read_csv(target_path)
        if "id" in df_target.columns:
            df_target = df_target.rename(columns={"id": "source"})

        LOGGER.info("Merging link prediction data with node features")
        out_df = link_prediction_df.merge(df_target, on="source", how="left")
    else:
        LOGGER.warning(
            "Node features file %s not found; saving link prediction dataset without features",
            target_path,
        )
        out_df = link_prediction_df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[Strategy] Completed dataset to {out_path}")
    LOGGER.info("Saved processed dataset to %s", out_path)


if __name__ == "__main__":
    strategy()
