# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle as pk
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import export_graphviz
from util.log import *
from sklearn import tree


def accuracy(policy, obss, acts):
    return np.mean(np.all(acts == policy.predict(obss), axis=1))


def split_train_test(obss, acts, train_frac, return_index=False):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    if not return_index:
        return obss_train, acts_train, obss_test, acts_test
    else:
        return obss_train, acts_train, obss_test, acts_test, idx[:n_train], idx[n_train:]


def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()


def save_dt_policy_viz(dt_policy, feature_names, class_names, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    export_graphviz(dt_policy.tree_root,
                    filled=True,
                    # feature_names=feature_names,
                    class_names=class_names,
                    out_file=dirname + '/' + fname)
    print(tree.export_text(dt_policy.tree_root))


def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy


class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        train_accuracy = accuracy(self, obss_train, acts_train)
        test_accuracy = accuracy(self, obss_test, acts_test)
        num_nodes = self.tree.tree_.node_count
        feature_importances = self.tree.feature_importances_
        log('Train accuracy: {}'.format(accuracy(self, obss_train, acts_train)), INFO)
        log('Test accuracy: {}'.format(accuracy(self, obss_test, acts_test)), INFO)
        log('Number of nodes: {}'.format(self.tree.tree_.node_count), INFO)
        return_tuple = [train_accuracy, test_accuracy, num_nodes, feature_importances]
        return return_tuple

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone


class RandomForestPolicy:
    def __init__(self, max_depth, n_estimators=4):
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def fit(self, obss, acts):
        self.trees = BaggingClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators
        )
        self.trees.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        train_accuracies = [
            accuracy(self.trees.estimators[i], obss_train, acts_train) for i in range(self.n_estimators)
        ]
        test_accuracies = [
            accuracy(self.trees.estimators[i], obss_test, acts_test) for i in range(self.n_estimators)
        ]
        feature_importances = [
            estimator.tree.feature_importances_ for estimator in self.trees.estimators
        ]
        num_nodes = [
            estimator.tree.tree_.node_count for estimator in self.trees.estimators
        ]
        log('Train accuracies: {}'.format(train_accuracies))
        log('Test accuracies: {}'.format(test_accuracies))
        log('Number of ndoes: {}'.format(num_nodes))
        return_tuple = [
            train_accuracies, test_accuracies, num_nodes, feature_importances
        ]
        return return_tuple

    def mask_features(self, obss):
        # TODO: implement feature masking 
        pass

    def predict(self, obss):
        predictions = [
            estimator.tree.predict(obss) for estimator in self.trees.estimators
        ]
        return predictions

    def clone(self):
        clone = BaggingClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators)
        for i, clon in enumerate(clone.estimators):
            clon.tree = self.trees.estimators[i].tree
        return clone
