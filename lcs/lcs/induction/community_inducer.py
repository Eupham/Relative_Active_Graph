"""
community_inducer.py — SBM community detection via Leiden + MDL.

Replaces k-means TRD crystallisation.

Algorithm:
  1. Build a weighted co-occurrence graph G from token surfaces.
     Nodes: unique surface forms. Edge weight: co-occurrence count within
     a sliding window over the training corpus.
  2. Run the Leiden algorithm to find an initial community partition.
  3. Evaluate the partition under Minimum Description Length:
       L = L(G | partition) + L(partition)
     where L(G | partition) is the SBM log-likelihood and L(partition)
     encodes the partition structure itself.
  4. Iterate Leiden until MDL stops decreasing.
  5. Return community assignments as TypeCategory IDs.

References:
  Traag et al. (2019), Peixoto (2014), Rissanen (1978).
"""
from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommunityRecord:
    """One discovered community (TypeCategory)."""
    id:       int             # opaque integer ID; no linguistic meaning
    members:  list[str]       # surface forms in this community
    top5:     list[str]       # 5 most frequent members (cluster descriptor)
    size:     int


class CommunityInducer:
    """
    Builds a co-occurrence graph and partitions it via Leiden + MDL.

    Usage:
        inducer = CommunityInducer(window=3)
        inducer.observe_sentence(tokens)  # repeated for all training sentences
        records = inducer.fit()           # returns list[CommunityRecord]
        cat_id  = inducer.assign(surface) # assign a new surface to a community
    """

    def __init__(self, window: int = 3) -> None:
        self.window = window
        self._cooc: dict[tuple[str, str], int] = defaultdict(int)
        self._freq: dict[str, int] = defaultdict(int)
        self._communities: list[CommunityRecord] = []
        self._assignment: dict[str, int] = {}   # surface → community ID

    def observe_sentence(self, tokens: list[str]) -> None:
        """Record co-occurrences within a sliding window."""
        for i, tok in enumerate(tokens):
            self._freq[tok] += 1
            for j in range(i + 1, min(i + self.window + 1, len(tokens))):
                pair = (min(tok, tokens[j]), max(tok, tokens[j]))
                self._cooc[pair] += 1

    def fit(self) -> list[CommunityRecord]:
        """
        Build the co-occurrence graph and partition it.
        Returns list[CommunityRecord].
        """
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError(
                "leidenalg and igraph are required for community induction. "
                "pip install leidenalg python-igraph"
            )

        if not self._cooc:
            logger.warning("No co-occurrence data observed; returning empty partition.")
            return []

        # Build igraph weighted graph.
        surfaces = sorted(set(
            s for pair in self._cooc for s in pair
        ))
        idx = {s: i for i, s in enumerate(surfaces)}
        edges     = [(idx[a], idx[b]) for (a, b) in self._cooc]
        weights   = [float(w) for w in self._cooc.values()]
        G = ig.Graph(n=len(surfaces), edges=edges, directed=False)
        G.es["weight"] = weights

        # Run Leiden with MDL-guided partition selection.
        best_partition = None
        best_mdl       = float("inf")
        # Try multiple resolution parameters; MDL selects the best.
        for resolution in [0.1, 0.5, 1.0, 2.0, 5.0]:
            partition = leidenalg.find_partition(
                G,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                n_iterations=10,
                seed=42,
            )
            mdl = self._mdl(G, partition, weights)
            if mdl < best_mdl:
                best_mdl       = mdl
                best_partition = partition

        # Build CommunityRecord list from best partition.
        self._communities = []
        self._assignment  = {}
        for comm_id, members_idx in enumerate(best_partition):
            members = [surfaces[i] for i in members_idx]
            top5    = sorted(members,
                             key=lambda s: self._freq.get(s, 0),
                             reverse=True)[:5]
            for s in members:
                self._assignment[s] = comm_id + 1  # IDs start at 1
            self._communities.append(CommunityRecord(
                id=comm_id + 1,
                members=members,
                top5=top5,
                size=len(members),
            ))

        logger.info("SBM Leiden MDL: %d communities (MDL=%.2f)", len(self._communities), best_mdl)
        return self._communities

    def assign(self, surface: str) -> int:
        """
        Return the community ID for a surface form.
        New surfaces (not seen during fit) are assigned via majority vote
        over the co-occurrence graph neighborhood.
        Returns 0 (DEFAULT) when no assignment can be made.
        """
        if surface in self._assignment:
            return self._assignment[surface]
        # Majority vote over neighbors observed during training.
        votes: dict[int, int] = defaultdict(int)
        for (a, b), w in self._cooc.items():
            neighbor = b if a == surface else (a if b == surface else None)
            if neighbor and neighbor in self._assignment:
                votes[self._assignment[neighbor]] += w
        if votes:
            return max(votes, key=votes.__getitem__)
        return 0  # DEFAULT

    # ── MDL criterion ─────────────────────────────────────────────────────────

    def _mdl(self, G, partition, weights: list[float]) -> float:
        """
        Minimum Description Length for a partition of G.
        L = L(G | partition) + L(partition)

        L(G | partition) = -log P(G | SBM parameters) using the microcanonical SBM.
        L(partition)     = log of the number of ways to assign n nodes into B blocks.

        Reference: Peixoto (2014) §II.
        """
        n = G.vcount()
        B = len(set(partition.membership))

        # L(partition): encoding cost of the partition itself.
        # Stirling approximation for log C(n, B) + B * log n.
        l_partition = B * math.log(n + 1) if n > 0 else 0.0

        # L(G | partition): negative log-likelihood under planted partition model.
        # Approximation: sum over blocks of -e_rs * log(e_rs / (n_r * n_s))
        # where e_rs = edge count between blocks r and s.
        membership = partition.membership
        block_sizes: dict[int, int] = defaultdict(int)
        for m in membership:
            block_sizes[m] += 1

        edge_counts: dict[tuple[int, int], float] = defaultdict(float)
        for (u, v), w in zip(G.get_edgelist(), weights):
            r, s = membership[u], membership[v]
            key = (min(r, s), max(r, s))
            edge_counts[key] += w

        l_graph = 0.0
        for (r, s), e_rs in edge_counts.items():
            if e_rs <= 0:
                continue
            n_r = block_sizes[r]
            n_s = block_sizes[s] if s != r else block_sizes[r]
            denom = n_r * n_s
            if denom > 0:
                l_graph -= e_rs * math.log(e_rs / denom)

        return l_graph + l_partition
