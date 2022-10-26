import os
import random
import time
from copy import copy

import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sortedcontainers import SortedList

from maviper.python.viper.core.dt import split_train_test
import numpy as np
from util.log import *
from tqdm import tqdm

from maviper.python.viper.util.log import log, INFO
from utils.make_env import make_env


class JointDTPolicy:
    class Node:
        def __init__(self, gini, samples, num_samples, focus_samples, num_samples_per_class, most_frequent_class, index,
                     cur_depth,
                     max_depth, joint=True):
            self.gini = gini
            self.samples = copy(samples)
            self.num_samples = num_samples
            self.focus_samples = copy(focus_samples)
            self.num_samples_per_class = num_samples_per_class
            self.most_frequent_class = most_frequent_class
            self.feature_index = None
            self.threshold = 0
            self.left = None
            self.right = None
            self.index = index
            self.updated = True
            self.max_depth, self.cur_depth = max_depth, cur_depth
            # Create hash index
            self.hash = set([x.tostring() for x in self.samples[0][index]])
            self.joint = joint
            self.update()

        def add_sample(self, x, y):
            # Check to make sure not already present
            if x[self.index].tostring() in self.hash:
                return
            for i in range(len(x)):
                self.samples[0][i] = np.concatenate((self.samples[0][i], x[i].reshape(1, -1)))
                self.samples[1][i] = np.concatenate((self.samples[1][i], y[i].reshape(-1)))
            self.num_samples += 1
            all_y = self.samples[1][self.index]
            self.num_samples_per_class = [np.sum(all_y == i) for i in range(len(self.num_samples_per_class))]
            self.most_frequent_class = np.argmax(self.num_samples_per_class)
            self.gini = 1.0 - sum((c / self.num_samples) ** 2 for c in self.num_samples_per_class)
            self.updated = True

        def update(self):
            if not self.updated:
                return
            self.updated = False
            X, y = self.samples[0][self.index], self.samples[1][self.index]
            if self.cur_depth >= self.max_depth or not self.joint:
                self.dt = None
                return
            self.dt = DecisionTreeClassifier(max_depth=max(1, self.max_depth - self.cur_depth)).fit(X, y)

        def get_decision_tree(self):
            return self.dt

        def remove_data(self):
            self.samples = self.focus_samples = self.dt = self.hash = None

    class Individual:
        def __init__(self, root, num_action, remove_data=True):
            self.root = copy(root)
            self.num_action = num_action
            if not remove_data:
                return

            # remove data
            queue = [self.root]
            while len(queue) > 0:
                node = queue.pop(0)
                node.remove_data()
                if node.left is not None:
                    node.left = copy(node.left)
                    queue.append(node.left)
                if node.right is not None:
                    node.right = copy(node.right)
                    queue.append(node.right)

        def traverse(self, obs):
            cur_node = self.root
            while cur_node is not None and cur_node.left is not None and cur_node.right is not None:
                if obs[cur_node.feature_index] < cur_node.threshold:
                    cur_node = cur_node.left
                else:
                    cur_node = cur_node.right
            return cur_node

        def predict(self, obss):
            ret = []
            for obs in obss:
                node = self.traverse(obs)
                ret.append(np.eye(self.num_action)[node.most_frequent_class])
            return np.array(ret)

        def clone(self):
            return copy(self)

        def measure_feature_importance(self):
            queue = [self.root]
            importance_data = dict()
            while len(queue) > 0:
                node = queue.pop(0)
                if node.left is not None and node.right is not None:
                    if node.feature_index not in importance_data:
                        importance_data[node.feature_index] = 0
                    importance_data[
                        node.feature_index] += node.num_samples * node.gini - node.left.num_samples * node.left.gini - node.right.num_samples * node.right.gini
                if node.left is not None:
                    node.left = copy(node.left)
                    queue.append(node.left)
                if node.right is not None:
                    node.right = copy(node.right)
                    queue.append(node.right)
            for i, v in importance_data.items():
                importance_data[i] = v / self.root.num_samples
            return importance_data

        @staticmethod
        def is_leaf(node):
            return node.left is None and node.right is None

        def visualize(self, save_path):
            # Set parent
            queue = [self.root]
            parent = dict()
            while len(queue) > 0:
                node = queue.pop(0)
                if node.left is not None:
                    parent[node.left] = node
                    queue.append(node.left)
                if node.right is not None:
                    parent[node.right] = node
                    queue.append(node.right)

            dot_output = []
            queue = [self.root]
            cnt_dict = dict()
            color_dict = {
                0: '#fcb71180',
                1: '#f3702180',
                2: '#cc004c80',
                3: '#6460aa80',
                4: '#0089d080',
                5: '#0db14b80'
            }
            action_dict = {
                0: 'STAY',
                1: 'UP',
                2: 'DOWN',
                3: 'LEFT',
                4: 'RIGHT'
            }
            while len(queue) > 0:
                node = queue.pop(0)
                threshold = "%.2f" % node.threshold
                cnt_dict[node] = len(cnt_dict)
                if self.is_leaf(node):
                    text = f'{cnt_dict[node]} [label="{action_dict[node.most_frequent_class]}", shape=box, color="{color_dict[node.most_frequent_class]}", style=filled,fontname="Arial"] ;'
                    dot_output.append(text)
                else:
                    state = f"State_{node.feature_index}"
                    text = f'{cnt_dict[node]} [label="{state} \n < {threshold}", color="#6460AA", shape=box,fontname="Arial"] ;'
                    dot_output.append(text)
                if node in parent:
                    if parent[node].left == node:
                        link_to_parent = f'{cnt_dict[parent[node]]} -> {cnt_dict[node]} [label="True", arrowsize=0.8, color="#8B8C89",fontname="Arial", angle=30] ;'
                    else:
                        link_to_parent = f'{cnt_dict[parent[node]]} -> {cnt_dict[node]} [label="False", arrowsize=0.8, color="#8B8C89",fontname="Arial", angle=30] ;'
                    dot_output.append(link_to_parent)
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)

            output = "digraph Tree {\n" + "\n".join(dot_output) + "\n}"
            save_path = save_path.replace('results', 'visualization_results')
            os.makedirs(save_path[:save_path.rfind('/')], exist_ok=True)

            with open(f'{save_path}.tmp', 'w') as f:
                f.write(output)

            from subprocess import check_call
            check_call(['dot', '-Tpng', '-Gdpi=800', f'{save_path}.tmp', '-o', f'{save_path}.png'])
            check_call(['rm', f'{save_path}.tmp'])

    def make_individual(self):
        return [self.Individual(root, self._n_classes[i]) for i, root in enumerate(self.tree_root)]

    def __init__(self, max_depth, env, num_policies, team_info=None):
        self.max_depth = max_depth
        self.env = env
        self.num_policies = num_policies
        self.tree_root, self.dt = [None] * num_policies, [None] * num_policies
        self.add = [[] for _ in range(num_policies)]
        self.all_nodes = []
        self.all_training_samples = []

        # Set up teams_
        assert team_info is not None, "forgot to pass in a valid team setup?"
        self.agent_team = dict()
        self.team_setup = [list() for _ in range(self.num_policies)]
        for agent in range(self.num_policies):
            self.agent_team[agent] = team_info[agent]
            self.team_setup[self.agent_team[agent]].append(agent)

    def __clean(self):
        queue = [self.tree_root[i] for i in range(self.num_policies)]
        while len(queue) > 0:
            node = queue.pop(0)
            if node is None:
                continue
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
            node.remove_data()
        self.all_training_samples = None
        self.tree_root = [None] * self.num_policies
        self.all_nodes = []

    def _gini(self, status, hard=True, y=None, return_status=False, add=True, add_all=False, changes_dict=None):
        if hard:
            return 1.0 - sum((c / sum(status)) ** 2 for c in status)
        else:
            score = self._gini(status, hard=True)
            if return_status:
                return score, status, None
            else:
                return score, None

    def _get_best_split(self, tree_index, X, y, q=None):
        num_samples = len(y)
        if num_samples < 2:  # No need to split
            return None, None
        num_samples_per_class = [np.sum(y == i) for i in range(self._n_classes[tree_index])]

        best_gini = self._gini(num_samples_per_class)
        best_feature, best_threshold, best_status = None, None, None

        num_feature = X.shape[1]
        fn = lambda g, x: g

        for index in range(num_feature):
            sorted_feature = sorted(zip(X[:, index], y))
            status_left, status_right = [0] * self._n_classes[tree_index], num_samples_per_class.copy()

            for i in range(1, num_samples):
                c = sorted_feature[i - 1][1]
                status_left[c] += 1
                status_right[c] -= 1

                gini_left = self._gini(status_left, hard=True)
                gini_right = self._gini(status_right, hard=True)

                cur_gini = (i * gini_left + (num_samples - i) * gini_right) / num_samples
                if sorted_feature[i][0] != sorted_feature[i - 1][0] and fn(cur_gini, index) < best_gini:
                    best_gini = fn(cur_gini, index)
                    best_feature = index
                    best_threshold = (sorted_feature[i][0] + sorted_feature[i - 1][0]) / 2
        return best_feature, best_threshold

    def _traverse(self, x, index):
        cur_node = self.tree_root[index]
        cur_depth = 0
        while cur_node is not None and cur_node.left is not None and cur_node.right is not None:
            cur_depth += 1
            if x[cur_node.feature_index] < cur_node.threshold:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        return cur_node, cur_depth

    def _evaluate(self, x, y_correct, index, store):
        if not self.joint:
            return True
        cur_node, _ = self._traverse(x, index)

        if cur_node is None:
            dt = self.dt[index]
        else:
            dt = cur_node.get_decision_tree()

        if dt is not None:
            # Lazy prediction
            if dt not in self.lazy_predict:
                self.lazy_predict[dt] = list()
                self.lazy_target[dt] = list()
                self.lazy_store[dt] = list()

            self.lazy_predict[dt].append(x)
            self.lazy_target[dt].append(y_correct)
            self.lazy_store[dt].append(store)
        else:
            y_predict = cur_node.most_frequent_class
            self.all_votes[store].append(y_predict == y_correct)

    def _gather(self, l, ind):
        ret = []
        for i in range(len(l)):
            ret.append(l[i][ind])
        return ret

    def _build_tree(self, X, y, index, depth, recur_limit, parent, side):
        num_samples = len(y[index])
        if num_samples < 2:
            return None

        # Check if tree is already built here
        # Skip the querying process if already done before (since trees are built layer by layer)
        if parent is not None and parent.left is not None and parent.right is not None:
            if side == -1:
                node = parent  # root node
            else:
                node = parent.left if side == 0 else parent.right
            all_X, all_y = node.samples
            X, y = node.focus_samples
        else:
            focus_X, focus_y, focus_q = [], [], []
            all_X, all_y = [[] for _ in range(self.num_policies)], [[] for _ in range(self.num_policies)]

            self.all_votes = [list() for _ in range(len(X[index]))]
            self.lazy_predict, self.lazy_target, self.lazy_store = dict(), dict(), dict()
            for k in range(len(X[index])):
                for i in self.team_setup[self.agent_team[index]]:
                    self._evaluate(X[i][k], y[i][k], index=i, store=k)
            for dt, to_predict in self.lazy_predict.items():
                predict = dt.predict(np.stack(to_predict, axis=0))
                correct = predict == np.stack(self.lazy_target[dt], axis=0)
                for i, s in enumerate(self.lazy_store[dt]):
                    self.all_votes[s].append(correct[i])

            for k in range(len(X[index])):
                vote = self.all_votes[k]

                add = sum(vote) > len(
                    self.team_setup[self.agent_team[index]]) * self.parameters.threshold or not self.joint

                if add:
                    for i in range(self.num_policies):
                        all_X[i].append(X[i][k])
                        all_y[i].append(y[i][k])

                    focus_X.append(X[index][k])
                    focus_y.append(y[index][k])

            if len(focus_y) == 0:  # if no data is left, then we just use the entire dataset
                focus_X, focus_y = X[index], y[index]
                all_X, all_y = X, y

            X, y = np.array(focus_X), np.array(focus_y)
            for i in range(self.num_policies):
                all_X[i] = np.array(all_X[i])
                all_y[i] = np.array(all_y[i])

            gini, num_samples_per_class, _ = self._gini([np.sum(y == i) for i in range(self._n_classes[index])],
                                                        hard=False,
                                                        return_status=True)

            node = JointDTPolicy.Node(gini=gini,
                                      samples=(all_X, all_y),
                                      num_samples=len(all_y[index]),
                                      focus_samples=(X, y),
                                      num_samples_per_class=num_samples_per_class,
                                      most_frequent_class=np.argmax(num_samples_per_class),
                                      index=index,
                                      cur_depth=depth,
                                      max_depth=self.max_depth,
                                      joint=self.joint
                                      )
            self.all_nodes.append(node)

        # Split recursively
        if depth < min(self.max_depth, recur_limit):
            # Check if already split
            if node.feature_index is not None:
                feature_index, feature_threshold = node.feature_index, node.threshold
            else:
                feature_index, feature_threshold = self._get_best_split(index, X, y)
            if feature_index is not None:
                indices_left = all_X[index][:, feature_index] < feature_threshold
                X_left, y_left = self._gather(all_X, indices_left), self._gather(all_y, indices_left)
                X_right, y_right = self._gather(all_X, ~indices_left), self._gather(all_y, ~indices_left)
                node.feature_index = feature_index
                node.threshold = feature_threshold
                node.left = self._build_tree(X_left, y_left, index, depth + 1, recur_limit, parent=node, side=0)
                node.right = self._build_tree(X_right, y_right, index, depth + 1, recur_limit, parent=node, side=1)
        return node

    def _parse(self, array, t):
        assert len(array.shape) == 2  # array: size x joint obs/act
        ret = []
        cnt = 0
        for i in range(self.num_policies):  # num_policies is equal to num of agents
            if t == 'obs':
                cur = self.env.observation_space[i].shape[0]
                if self.parameters.scenario_name == 'simple_tag' and self.parameters.easy_feature:
                    cur = self.env.easy_feature_sizes[i]
                ret.append(array[:, cnt:cnt + cur])
            else:
                cur = self.env.action_space[i].n
                if t == 'act':
                    ret.append(np.argmax(array[:, cnt:cnt + cur], axis=1))
                elif t == 'act_onehot':
                    assert len(array) == 1, 'this is only for predict() function'
                    ret.append(array[0, cnt:cnt + cur])
            cnt += cur
        return ret  # can't use concatenate due to different size

    def _fit(self, samples):
        # Build new trees
        self.dt = []
        for i in range(self.num_policies):
            self.tree_root[i] = None
            X, y = self.all_training_samples[i][0][i], self.all_training_samples[i][1][i]
            # Initialize decision tree for prediction
            self.dt.append(DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y))

        # Build trees layer by layer, i.e., by the order -- 1st layer for DT1, 1st layer for DT2
        for depth in range(0, self.max_depth + 1):
            indices = list(range(self.num_policies))
            random.shuffle(indices)
            for i in indices:
                self.tree_root[i] = self._build_tree(samples[i][0], samples[i][1], index=i, depth=0,
                                                     recur_limit=depth,
                                                     parent=self.tree_root[i], side=-1)
                for node in self.all_nodes:
                    node.update()

            if depth == 0:
                for i in range(self.num_policies):
                    for sample in self.add[i]:
                        self.tree_root[i].add_sample(*sample)
                    self.add[i].clear()

        # Train accuracy for each agent
        log(f'Train accuracy (joint): {self.eval(self.all_training_samples, obs=0, act=1, joint_eval=True)}', INFO)
        log(f'Train accuracy (individual): {self.eval(self.all_training_samples, obs=0, act=1, joint_eval=False)}',
            INFO)
        log(f'Test accuracy (joint): {self.eval(self.all_training_samples, obs=2, act=3, joint_eval=True)}', INFO)
        log(f'Test accuracy (individual): {self.eval(self.all_training_samples, obs=2, act=3, joint_eval=False)}', INFO)

    def fit(self, samples):
        samples = copy(samples)
        self._fit(samples)

    def eval(self, data, obs, act, joint_eval):
        ret = []
        for i in range(self.num_policies):
            acts_predict = self._predict(data[i][obs])
            num_correct = 0
            for j in range(acts_predict.shape[0]):
                if joint_eval:
                    correct = [acts_predict[j][data[i][act][k][j] + sum(self._n_classes[:k])] for k in
                               self.team_setup[self.agent_team[i]]]
                else:
                    correct = [acts_predict[j][data[i][act][i][j] + sum(self._n_classes[:i])]]
                num_correct += int(np.sum(correct) == len(correct))
            ret.append(num_correct / acts_predict.shape[0])
        return ret

    def train(self, obss_train, acts_train, obss_test, acts_test, parameters=None):
        self.__clean()
        self.parameters = copy(parameters)
        self.joint = not parameters.not_joint
        # self.all_training_samples is stored as:
        # self.all_training_samples[id of the policy which the data is for][0 for obs, 1 for act][policy id of the partial obs/act]
        self._n_classes = [self.env.action_space[i].n for i in range(self.num_policies)]
        self.all_training_samples = [(self._parse(obss_train[i], t='obs'), self._parse(acts_train[i], t='act'),
                                      self._parse(obss_test[i], t='obs'), self._parse(acts_test[i], t='act')) for i in
                                     range(self.num_policies)]  # (policy_id, obs/act, partial for each policy)
        self.total_time = 0
        self.fit(self.all_training_samples)
        print(f'Total time: {self.total_time}')

    def _predict(self, obss):
        ret = []
        for k in range(obss[0].shape[0]):
            act = []
            for i in range(self.num_policies):
                cur_node, cur_depth = self._traverse(obss[i][k], index=i)
                act.append(np.eye(self._n_classes[i])[cur_node.most_frequent_class])
            act = np.concatenate(act)
            ret.append(act)
        return np.array(ret)

    def predict(self, obss):  # obss: size x #agent x obs_length
        new_obss = []
        for i in range(self.num_policies):
            new_obss.append(np.array([obss[j][i] for j in range(len(obss))]))
        return [self._parse(self._predict(new_obss), t='act_onehot')]
