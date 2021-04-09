import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def lab5(x_train, x_test, y_train, y_test):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)  # max_leaf_nodes=10, random_state=0
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Прогноз y на тестовых данных",y_pred)
    print("Tочность на тренировочной выборке: {:.3f}".format(clf.score(x_train, y_train)))
    print("Tочность на тестовой выборке: {:.3f}".format(clf.score(x_test, y_test)))

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # начать с идентификатора корневого узла (0) и его глубины (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i]))

    plt.figure(figsize=(18, 18))
    tree.plot_tree(clf, fontsize=6, feature_names=x_train.columns)
    plt.savefig('tree_high_dpi', dpi=150)
    plt.show()
