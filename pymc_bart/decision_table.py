# decision_table.py
import numpy as np
from numba import njit

from pymc_bart.split_rules import SplitRule, ContinuousSplitRule
from pymc_bart.tree import get_depth  # только для совместимости, не используется внутри

class DecisionTable:
    """
    Oblivious (CatBoost-style) decision table.
    Все узлы одного уровня используют одну и ту же фичу,
    но разные пороги. Дерево всегда сбалансированное (все листья на одной глубине).
    """
    __slots__ = ("depth", "features", "thresholds", "leaf_node_values", "leaf_node_counts", 
                 "idx_data_points_list", "split_rules")

    def __init__(self, depth=0, features=None, thresholds=None, leaf_node_values=None,
                 idx_data_points_list=None, split_rules=None):
        self.depth = depth
        self.features = features or []                     # list[int], длина = depth
        self.thresholds = thresholds or []                   # list[np.ndarray], каждый shape (2**d,)
        self.leaf_node_values = leaf_node_values or np.array([])  # shape (2**depth, shape)
        self.idx_data_points_list = idx_data_points_list or []   # list[np.ndarray]
        self.split_rules = split_rules or []

    def copy(self):
        return DecisionTable(
            depth=self.depth,
            features=self.features.copy(),
            thresholds=[t.copy() for t in self.thresholds],
            leaf_node_values=self.leaf_values.copy() if self.leaf_node_values.size else None,
            idx_data_points_list=[idx.copy() for idx in self.idx_data_points_list],
            split_rules=self.split_rules,
        )

    def trim(self):
        """Убираем idx_data_points – нужно только для сэмплирования, не для предсказания"""
        return DecisionTable(
            depth=self.depth,
            features=self.features.copy(),
            thresholds=[t.copy() for t in self.thresholds],
            leaf_node_values=self.leaf_node_values.copy() if self.leaf_node_values.size else None,
            split_rules=self.split_rules,
        )

    def _predict(self, X):
        n = X.shape[0]
        if self.depth == 0:
            return np.full((n, self.leaf_node_values.shape[-1]), self.leaf_node_values)
        
        pred = np.zeros((n, self.leaf_node_values.shape[-1]))
        node_idx = np.zeros(n, dtype=np.int32)

        for d in range(self.depth):
            feat = self.features[d]
            thr = self.thresholds[d][node_idx]      # векторизовано по объектам
            left = X[np.arange(n), feat] <= thr
            node_idx[left] = 2 * node_idx[left] + 1
            node_idx[~left] = 2 * node_idx[~left] + 2

        pred = self.leaf_node_values[node_idx]
        return pred

    def get_leaf_nodes(self):
        """Возвращает список массивов с индексами точек в каждом листе"""
        return self.idx_data_points_list

    def __repr__(self):
        return f"DecisionTable(depth={self.depth}, leaves={2**self.depth if self.depth > 0 else 1})"
