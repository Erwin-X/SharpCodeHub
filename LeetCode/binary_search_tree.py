# BST的insert和traverse
from collections import deque

class TreeNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, key, val):
        if self.root is None:
            self.root = TreeNode(key, val)
        else:
            self._recursive_insert(self.root, key, val)
    def _recursive_insert(self, node, key, val):
        if key<=node.key:
            if node.left is None:
                node.left = TreeNode(key, val)
            else:
                self._recursive_insert(node.left, key, val)
        else:
            if node.right is None:
                node.right = TreeNode(key, val)
            else:
                self._recursive_insert(node.right, key, val)
    def level_order_traverse(self):
        if self.root is None:
            return []
        
        result = []
        queue = deque([self.root])
        while queue:
            cur_node = queue.popleft()
            result.append((cur_node.key, cur_node.val))
            
            if cur_node.left:
                queue.append(cur_node.left)
            if cur_node.right:
                queue.append(cur_node.right)
        return result
    def pre_order_traverse(self, node, result):
        if node is None:
            return []
        result.append((node.key, node.val))
        if node.left:
            self.pre_order_traverse(node.left, result)
        if node.right:
            self.pre_order_traverse(node.right, result)
        return result
    def mid_order_traverse(self, node, result):
        if node is None:
            return []
        if node.left:
            self.mid_order_traverse(node.left, result)
        result.append((node.key, node.val))
        if node.right:
            self.mid_order_traverse(node.right, result)
        return result
    def search(self, node, key):
        if node is None:
            return None
        if key == node.key:
            return (node.key, node.val)
        elif key < node.key:
            return self.search(node.left, key)
        else:
            return self.search(node.right, key)


if __name__ == "__main__":
    bst = BinarySearchTree()
    bst.insert(3, 'Three')
    bst.insert(1, "One")
    bst.insert(4, "Four")
    bst.insert(2, "Two")
    bst.insert(5, "Five")

    level_order_traverse = bst.level_order_traverse()
    print(level_order_traverse)

    pre_order_traverse = bst.pre_order_traverse(bst.root, [])
    print(pre_order_traverse)

    mid_order_traverse = bst.mid_order_traverse(bst.root, [])
    print(mid_order_traverse)

    for key in [2, 4, 5, 10]:
        print(bst.search(bst.root, key))
