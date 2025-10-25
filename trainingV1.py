import numpy as np

class TNode:
    def __init__(self):
        self.index = None    
        self.value = None      
        self.prediction = None 
        self.left_child = None 
        self.right_child = None

def find_best_split(x, y, min_samples_leaf=1):
    X = np.array(x)
    Y = np.array(y)
    num_samples, num_features = np.shape(X)

    best_feature = None
    best_threshold = None
    best_mse = float('inf')

    for i in range(num_features):
        order = np.argsort(X[:, i])
        xs = X[order, i]
        ys = Y[order]

        for j in range(num_samples - 1):
            ls = ys[:j+1]
            rs = ys[j+1:]

            if len(ls) < min_samples_leaf or len(rs) < min_samples_leaf:
                continue
            if xs[j] == xs[j + 1]:
                continue

            threshold = (xs[j] + xs[j + 1]) / 2
            meanl = np.mean(ls)
            meanr = np.mean(rs)

            varl = np.mean(ls**2) - meanl**2
            varr = np.mean(rs**2) - meanr**2

            n_left = len(ls)
            n_right = len(rs)

            mse = (n_left * varl + n_right * varr) / (n_left + n_right)

            if mse < best_mse:
                best_mse = mse
                best_threshold = threshold
                best_feature = i

    return best_feature, best_threshold, best_mse

def build_tree(x, y, min_samples_split=2, min_samples_leaf=1, depth=0, max_depth=None):
    X = np.array(x)
    Y = np.array(y)

    # stop condition: too few samples or depth limit
    if len(X) < min_samples_split or (max_depth is not None and depth >= max_depth):
        leaf = TNode()
        leaf.prediction = np.mean(Y)
        return leaf

    # find the best split
    best_feature, best_threshold, best_mse = find_best_split(X, Y, min_samples_leaf)

    # no valid split found
    if best_feature is None:
        leaf = TNode()
        leaf.prediction = np.mean(Y)
        return leaf

    # split data
    xl, yl, xr, yr = [], [], [], []
    for i in range(len(X)):
        if X[i, best_feature] <= best_threshold:
            xl.append(X[i])
            yl.append(Y[i])
        else:
            xr.append(X[i])
            yr.append(Y[i])

    # if split fails (empty side)
    if len(xl) == 0 or len(xr) == 0:
        leaf = TNode()
        leaf.prediction = np.mean(Y)
        return leaf

    # build node
    node = TNode()
    node.index = best_feature
    node.value = best_threshold
    node.prediction = np.mean(Y)
    node.left_child = build_tree(xl, yl, min_samples_split, min_samples_leaf, depth+1, max_depth)
    node.right_child = build_tree(xr, yr, min_samples_split, min_samples_leaf, depth+1, max_depth)

    return node
