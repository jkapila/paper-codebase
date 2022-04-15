import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class TreeNode:

    def __init__(self, parent, depth, func=None, agg_func=None, constrain='soft'):

        self.parent = parent
        self.depth = depth
        self.children = []
        self.value = []
        self.is_end = False
        self.constrain = constrain  # 'hard' or 'soft'

        if func is None:
            self.func = lambda x: np.sum(x)
        else:
            self.func = func

        if agg_func is None:
            self.agg_func = lambda x: np.sum(x)
        else:
            self.agg_func = func

    def split(self, img, resolution):
        x_n, y_n = img.shape
        x = (0, int(np.floor(x_n / 3)), x_n - int(np.floor(x_n / 3)), x_n)
        y = (0, int(np.floor(y_n / 3)), y_n - int(np.floor(y_n / 3)), y_n)
        i = 0
        split_array = []
        for x_i in range(4):
            for y_i in range(4):
                if x[x_i] != 0 and y[y_i] != 0:
                    sub_array = img[x[x_i - 1]:x[x_i], y[y_i - 1]:y[y_i]]
                    split_array.append(sub_array)
                    i += 1
        if len(split_array) < 9:
            self.value = self.func(split_array)
            self.is_end = True

        else:
            if self.constrain == 'hard':
                # even if one sub_array has reached resolution calculate func for all of them 
                cond = [True if i.shape[0] * i.shape[1] >= resolution else False for i in split_array]
                if sum(cond) < 9:
                    self.value = [self.func(i) for i in split_array]
                    self.is_end = True
                else:
                    self.value = split_array

            elif self.constrain == 'soft':
                # keep calculating on sub_arrays on which splitting is possible
                for i in split_array:
                    if i.shape[0] * i.shape[1] >= resolution:
                        self.value.append(i)
                    else:
                        self.value.append(self.func(i))
                if sum([True if np.array(i).shape == (1, 1) else False for i in self.value]) == 9:
                    self.is_end = True

    def recursive_splits(self, img, resolution):
        self.split(img, resolution)
        if len(self.value) == 9:
            for i in self.value:
                try:
                    if i.shape[0] * i.shape[1] >= resolution:
                        new_node = TreeNode(self, depth=(self.depth + 1))
                        new_node.recursive_splits(i, resolution)
                        self.children.append(new_node)
                #                     elif np.asarray(i).shape == (1,1):
                #                         self.children.append(None)                        
                except:
                    self.children.append(None)
        self.value = [self.func(i) if i.shape is not () else i for i in self.value]

    def __repr__(self):

        if self.depth == 0:
            spacer = '|~~'
        else:
            spacer = '|  ' * (self.depth) + '|~~'

        s = spacer + 'Node at depth {} has {} children and Value: \n{}{}\n'.format((self.depth + 1), sum([
            True if i is not None else False for i in self.children]), spacer,
                                                                                   [i.shape if i.shape is not () else i
                                                                                    for i in self.value])
        for i in self.children:
            if i is not None:
                s += i.__repr__()

        return s

    def expand(img, resolution):
        pass

    def collect(self, zoom):
        #         if not self.is_leaf:
        value = []
        for child in self.children:
            if child is not None:
                val_ = child.collect(zoom)
                value.append(val_)
        return value

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return sum([True if i is not None else False for i in self.children]) == 0

    def is_root(self):
        return self.parent is None


tree = TreeNode(None, depth=0, func=lambda x: round(np.max(x) - np.min(x), 2),
                agg_func=lambda x: 0,
                constrain='soft')


if __name__ == '__main__':
    # tree= TreeNode(None,0,constrain='hard')
    # tree.recursive_splits(np.random.random_sample((100,100)),2)
    img = np.round(np.random.random_sample((28, 28)), 2)
    # sns.heatmap(img)
    tree.recursive_splits(img, 4)
    # print(tree.value)
    tree.collect(1)