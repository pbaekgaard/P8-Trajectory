from typing import List, Tuple
import numpy as np
import math



class DDPTree(object):
    def __init__(self, value: float = None, encoded_string: str = None):
        super().__init__()
        self.left: DDPTree = None
        self.right: DDPTree = None
        self.value: float = value  # Float if black, None if white
        self.encoded_string: str = encoded_string #Fills up as DDPTree is filled.

    def append_leafs(self, left = None, right = None):
        self.left = left
        self.right = right

    def append_encoded_string(self, encoded_string: str):
        self.encoded_string = self.encoded_string + encoded_string



class TEDTrajectory(object):
    def __init__(self, entry_path: np.ndarray, time_flags: np.ndarray, time_seq: np.ndarray, distance_seq: np.ndarray):
        super().__init__()
        self.entry_path = entry_path
        self.time_flags = time_flags
        self.time_seq = time_seq
        self.distance_seq = distance_seq


class TEDCompressed(object):
    def __init__(self, compressed_entry_paths: np.ndarray, compressed_time_flags: np.ndarray,
                 compressed_time_seqs: np.ndarray, compressed_distance_seqs: np.ndarray):
        super().__init__()
        self.compressed_entry_paths = compressed_entry_paths
        self.compressed_time_flags = compressed_time_flags
        self.compressed_time_seqs = compressed_time_seqs
        self.compressed_distance_seqs = compressed_distance_seqs


class TEDCompressor(object):
    def __init__(self, k: int, n, m):
        super().__init__()
        self.k = math.floor(math.log2(k) + 1)
        self.shape = (n, m)

    def compress(self, ted_trajectories: List[TEDTrajectory]) -> TEDCompressed:
        M = np.empty(self.shape, dtype=int)
        entry_path_primes = np.empty(self.shape, dtype=str)

        for index, ted_trajectory in enumerate(ted_trajectories):
            # compression of entry_paths:
            M[index], entry_path_primes[index] = self.compress_entry_path(ted_trajectory.entry_path)
            # compression of distance seq:
            DDPTree = self.compress_distance_seq(ted_trajectory.distance_seq)
        A, B = self.compress_M(M)

        print(A, B, M)

    def compress_M(self, M: np.ndarray) -> Tuple[np.ndarray]:
        columns_with_one = np.any(M == 1, axis=0)
        B = np.zeros(M.shape[1], dtype=int)
        A = np.empty((M.shape[0], len(np.where(columns_with_one)[0])), dtype=int)

        index_A = 0
        A = A.T

        for (idx, col) in enumerate(M.T):
            if np.any(col == 1):
                B[idx] = 1
                A[index_A] = col
                index_A += 1

        A = A.T

        return A, B

    def compress_entry_path(self, entry_path: np.ndarray) -> Tuple[np.ndarray]:
        entry_path_prime = self.int_to_binary(entry_path)

        M_row = np.empty(np.shape(entry_path_prime), dtype=int)
        index = 0

        for entry in np.nditer(entry_path_prime):
            M_row[index] = str(entry)[0]
            entry_path_prime[index] = str(entry)[1:]
            index += 1

        return M_row, entry_path_prime

    def compress_distance_seq(self, distance_seq: np.ndarray) -> DDPTree:
        ddp_tree = DDPTree(None, "")
        for entry in distance_seq:
            if entry != np.nan:
                ddp_tree.update_tree()


    def update_dp_tree(self, distance: float, tree_root: DDPTree, depth: int):
        err_bound = 0.02
        if distance - tree_root.value <= err_bound:
            return tree_root
        elif distance <= alpha[depth]:
            pass
        elif distance >= alpha[depth]:
            pass




    def int_to_binary(self, entry_path):
        # Convert each integer to binary, removing the '0b' prefix and padding to k digits
        binary_entry_path = np.array([format(num, f'0{self.k}b') for num in np.nditer(entry_path)], dtype=str)
        return binary_entry_path


entry_path = np.array([4, 2, 2, 1, 0, 6])
time_flags = np.array([1, 0, 1, 1, 1, 1])
time_seq = np.array([0, np.nan, 90, 180, 270, 361])
distance_seq = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75])
example_trajectory = TEDTrajectory(entry_path, time_flags, time_seq, distance_seq)

entry_path2 = np.array([3, 4, 2, 1, 0, 6])
time_flags2 = np.array([1, 0, 1, 1, 1, 1])
time_seq2 = np.array([0, np.nan, 90, 180, 270, 361])
distance_seq2 = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75])
example_trajectory2 = TEDTrajectory(entry_path2, time_flags2, time_seq2, distance_seq2)

trajectories = [example_trajectory, example_trajectory2]

alpha = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024]

if __name__ == '__main__':
    print("Hello TED")
    # print(example_trajectory)
    ted = TEDCompressor(
        k=7,
        n=len(trajectories),
        m=len(entry_path)
    )
    ted.compress(trajectories)
