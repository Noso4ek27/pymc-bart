# dt_mh_sampler.py
import numpy as np
from pymc.step_methods.arraystep import ArrayStepShared
from pymc_bart.bart import preprocess_xy
from pymc_bart.utils import _sample_posterior
from pymc_bart.pgbart import compute_prior_probability, SampleSplittingVariable
from pymc_bart.split_rules import ContinuousSplitRule
from decision_table import DecisionTable

class MHDTBART(ArrayStepShared):
    name = "mhdtbart"
    generates_stats = True
    stats_dtypes_shapes = {"variable_inclusion": (object, []), "accept": (float, [])}

    def __init__(self, vars=None, max_depth=6, alpha=0.95, beta=2.0, split_prior=None, **kwargs):
        super().__init__(vars, [])
        bart_rv = self.model.values_to_rvs[self.value_vars[0]].owner.op
        
        self.X, self.Y = preprocess_xy(bart_rv.X, bart_rv.Y)
        self.m = bart_rv.m
        self.shape = bart_rv.shape if hasattr(bart_rv, "shape") else 1
        self.max_depth = max_depth
        self.prior_prob_leaf = compute_prior_probability(alpha, beta)

        self.split_prior = np.ones(self.X.shape[1]) if split_prior is None else np.asarray(split_prior)
        self.split_prior /= self.split_prior.sum()

        # одна таблица на дерево
        self.tables = [DecisionTable() for _ in range(self.m)]  
        self.sum_trees = np.zeros((self.Y.shape[0], self.shape))

        # инициализация листовых значений
        init_value = self.Y.mean() / self.m
        for table in self.tables:
            table.leaf_node_values = np.full(self.shape, init_value)
            table.idx_data_points_list = [np.arange(len(self.X))]
            self.sum_trees += table._predict(self.X)

    def astep(self, _):
        accept_count = 0
        vi = np.zeros(self.X.shape[1])

        for i in range(self.m):
            table = self.tables[i]
            sum_without_i = self.sum_trees - table._predict(self.X)

            # три типа мувов с равной вероятностью
            move = np.random.choice(["grow", "prune", "change"])

            if move == "grow" and table.depth < self.max_depth:
                new_table = self.propose_grow(table, sum_without_i)
            elif move == "prune" and table.depth > 0:
                new_table = self.propose_prune(table)
            else:
                new_table = self.propose_change(table, sum_without_i)

            if new_table is None:
                continue

            new_sum = sum_without_i + new_table._predict(self.X)
            logp_new = pm.logp(self.model.rvs_to_values[0], new_sum.ravel())
            logp_old = pm.logp(self.model.rvs_to_values[0], self.sum_trees.ravel())

            prior_ratio = self.tree_prior_ratio(table, new_table)
            accept_prob = np.exp(logp_new - logp_old + prior_ratio)

            if np.random.random() < accept_prob:
                self.tables[i] = new_table
                self.sum_trees = new_sum
                accept_count += 1
                vi += new_table.get_split_variables()  # если нужно

        stats = {"variable_inclusion": _encode_vi(vi.astype(int).tolist()), "accept": accept_count / self.m}
        return self.sum_trees, [stats]

    def propose_grow(self, table, sum_without_i):
        # выбираем фичу
        feat = np.random.choice(len(self.split_prior), p=self.split_prior)
        new_thresholds = []

        new_idx_lists = []
        new_leaf_values = []

        for leaf_idx_arr in table.idx_data_points_list:
            points = self.X[leaf_idx_arr]
            if len(points) <= 1:
                return None

            rule = ContinuousSplitRule()
            split_val = rule.get_split_value(points[:, feat])
            if split_val is None:
                return None

            left = points[:, feat] <= split_val
            if not (0 < left.sum() < len(points)):
                return None

            new_thresholds.append(split_val)
            new_idx_lists.extend([leaf_idx_arr[left], leaf_idx_arr[~left]])

            # предлагаем новые leaf values как в PGBART
            for idxs in [leaf_idx_arr[left], leaf_idx_arr[~left]]:
                mu = (self.Y[idxs] - sum_without_i[idxs]).mean() / self.m
                new_leaf_values.append(mu + np.random.normal(0, self.Y.std() / np.sqrt(self.m)))

        new_table = table.copy()
        new_table.depth += 1
        new_table.features.append(feat)
        new_table.thresholds.append(np.array(new_thresholds))
        new_table.leaf_node_values = np.array(new_leaf_values)
        new_table.idx_data_points_list = new_idx_lists
        return new_table

    def propose_prune(self, table):
        # простейший prune – усредняем два дочерних листа в один
        new_table = table.copy()
        new_table.depth -= 1
        new_table.features.pop()
        new_table.thresholds.pop()

        old_leaves = table.leaf_node_values
        n = len(old_leaves) // 2
        new_values = []
        new_idx = []
        for i in range(n):
            w1 = len(table.idx_data_points_list[2*i])
            w2 = len(table.idx_data_points_list[2*i+1])
            new_val = (w1 * old_leaves[2*i] + w2 * old_leaves[2*i+1]) / (w1 + w2)
            new_values.append(new_val)
            new_idx.append(np.concatenate([table.idx_data_points_list[2*i],
                                          table.idx_data_points_list[2*i+1]]))
        new_table.leaf_node_values = np.array(new_values)
        new_table.idx_data_points_list = new_idx
        return new_table

    def propose_change(self, table, sum_without_i):
        # меняем пороги на случайном уровне
        if table.depth == 0:
            return None
        level = np.random.randint(table.depth)
        feat = table.features[level]
        new_thresholds = table.thresholds[level].copy()

        for node in range(len(new_thresholds)):
            leaf_idx_arr = table.idx_data_points_list[node]
            points = self.X[leaf_idx_arr]
            rule = ContinuousSplitRule()
            new_thr = rule.get_split_value(points[:, feat])
            if new_thr is not None:
                new_thresholds[node] = new_thr

        new_table = table.copy()
        new_table.thresholds[level] = new_thresholds
        # пересчитываем leaf values так же как в grow
        # (можно упростить)
        return new_table

    def tree_prior_ratio(self, old, new):
        # простейший вариант: log(p(new) - log(old) по глубине
        if new.depth > old.depth:
            return np.log(self.prior_prob_leaf[new.depth-1]) - np.log(1 - self.prior_prob_leaf[old.depth-1])
        elif new.depth < old.depth:
            return np.log(1 - self.prior_prob_leaf[new.depth]) - np.log(self.prior_prob_leaf[old.depth-1])
        return 0.0
