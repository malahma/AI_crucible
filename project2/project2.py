import numpy as np
import time

class SpatialKDTree:
    def __init__(self, data):
       
        self.tree = self._build(data)

    class Node:
        def __init__(self, point, axis, left=None, right=None):
            self.point = point
            self.axis = axis
            self.left = left
            self.right = right

    def _build(self, points, depth=0):
        n = len(points)
        if n == 0:
            return None


        axis = depth % 2
        
        points = points[points[:, axis].argsort()]
        median = n // 2

        return self.Node(
            point=points[median],
            axis=axis,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1:], depth + 1)
        )

    def query_nearest(self, target):
       
        self.best_point = None
        self.best_dist = float('inf')
        self._search(self.tree, np.array(target))
        return self.best_point, np.sqrt(self.best_dist)

    def _search(self, node, target):
        if node is None:
            return

        d_sq = np.sum((target - node.point) ** 2)

        if d_sq < self.best_dist:
            self.best_dist = d_sq
            self.best_point = node.point

        axis = node.axis
        diff = target[axis] - node.point[axis]

   
        near = node.left if diff < 0 else node.right
        far = node.right if diff < 0 else node.left

        self._search(near, target)

        if (diff ** 2) < self.best_dist:
            self._search(far, target)


if __name__ == "__main__":
    print("Initializing 100,000 NPC Coordinates")
    npc_coords = np.random.uniform(0, 1000, (100000, 2))
    player_tar = np.array([500.0, 500.0])

    print("Partitioning Map into KD-Tree")
    t0 = time.time()
    spatial_eng = SpatialKDTree(npc_coords)
    t1 = time.time()
    print(f"Tree built in: {t1 - t0:.4f}s")

    print(f"Querying Nearest Neighbor for {player_tar}")
    t2 = time.time()
    nearest_npc, distance = spatial_eng.query_nearest(player_tar)
    t3 = time.time()

   
    print(f"RESULT: NPC located at {nearest_npc}")
    print(f"DISTANCE: {distance:.4f} units")
    print(f"EXECUTION TIME: {t3 - t2:.6f} seconds")
 

    if (t3 - t2) < 0.1:
        print("Benchmark Passed")
    else:
        print("Benchmark Failed")