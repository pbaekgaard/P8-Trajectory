from typing import List, Tuple
import numpy as np
import math



class BinaryEncodingTree(object):
    def __init__(self, value: float = None, encoded_string: str = None):
        super().__init__()
        self.left: BinaryEncodingTree | None = None
        self.right: BinaryEncodingTree | None = None
        self.value: float | None = value  # Float if black, None if white

    def append_leafs(self, left = None, right = None):
        self.left = left or self.left # Update left tree if a new left tree is given otherwise keep the existing left tree.
        self.right = right or self.right # Update self.right if right is defined.
        return self

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
    def __init__(self, k: int, num_trajectories, num_entry_paths):
        super().__init__()
        self.k = math.floor(math.log2(k) + 1)
        self.shape = (num_trajectories, num_entry_paths)
        self.err_bound = 0.02

    def compress(self, ted_trajectories: List[TEDTrajectory]) -> TEDCompressed:
        M = np.empty(self.shape, dtype=int)
        entry_path_primes = np.empty(self.shape, dtype=str)
        pddp_tree_vector = np.empty((self.shape[0]), dtype=BinaryEncodingTree)  # een pr trajectory
        encoded_string_vector = np.empty((self.shape[0]), dtype=str)  # een pr trajectory
        T_prime_matrix = np.empty((self.shape[0]), dtype=object)  # GIGA MATRICE

        for index, ted_trajectory in enumerate(ted_trajectories):
            # compression of entry_paths:
            M[index], entry_path_primes[index] = self.compress_entry_path(ted_trajectory.entry_path)

            # compression of distance seq:
            pddp_tree_vector[index], encoded_string_vector[index] = self.compress_distance_seq(ted_trajectory.distance_seq)

            # compression of time_seq
            T_prime_matrix[index] = self.compress_time_seq(ted_trajectory.time_seq)

        A, B = self.compress_M(M) # GIGA MATRICE

        print("hello")



    def compress_time_seq(self, time_seq: np.ndarray) -> np.ndarray:
        time_seq = time_seq[~np.isnan(time_seq)]
        T_prime = np.full(np.shape(time_seq), None, dtype=object)
        last_saved_time_interval = None

        for index in range(0, len(T_prime) - 2):
            interval_one = time_seq[index + 1] - time_seq[index]
            interval_two = time_seq[index + 2] - time_seq[index + 1]

            if interval_two == interval_one:
                if last_saved_time_interval == interval_one:
                    continue
                else:
                    last_saved_time_interval = interval_one
                    T_prime[index] = (index, time_seq[index])
                    T_prime[index + 1] = (index + 1, time_seq[index + 1])
            else:
                if last_saved_time_interval is None:
                    T_prime[index] = (index, time_seq[index])
                else:
                    T_prime[index + 2] = (index + 2, time_seq[index + 2])
        return T_prime[T_prime != None] # Ignore suggestion from linter / IDE



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

    def compress_distance_seq(self, distance_seq: np.ndarray) -> Tuple[BinaryEncodingTree, str]:
        dp_tree = BinaryEncodingTree(None, "")
        for entry in distance_seq:
            if not np.isnan(entry):
                self.populate_dp_tree(distance=entry, sub_tree=dp_tree)

        ddp_tree = self.dp_to_ddp_tree(dp_tree)
        pddp_tree = self.dpp_to_pddp_tree(ddp_tree)

        encoded_string = ""

        for entry in distance_seq:
            if not np.isnan(entry):
                encoded_string += self.encode_pddp_tree(distance=entry, pddp_tree=ddp_tree)

        return pddp_tree, encoded_string


    def encode_pddp_tree(self, distance: float, pddp_tree: BinaryEncodingTree, encoded_string: str = "", tree_sum: float = 0, depth: int = 0) -> str:
        alpha_at_depth = alpha[depth]
        new_depth = depth + 1

        if pddp_tree.value is not None: return encoded_string

        if distance >= alpha_at_depth + tree_sum - self.err_bound: # Right
            encoded_string += "1"
            encoded_string = self.encode_pddp_tree(distance=distance, pddp_tree=pddp_tree.right, encoded_string=encoded_string, tree_sum=tree_sum+alpha_at_depth, depth=new_depth)
        else: # Left
            encoded_string += "0"
            encoded_string = self.encode_pddp_tree(distance=distance, pddp_tree=pddp_tree.left, encoded_string=encoded_string, tree_sum=tree_sum, depth=new_depth)

        return encoded_string

    def dpp_to_pddp_tree(self, ddp_tree: BinaryEncodingTree) -> BinaryEncodingTree:
        if ddp_tree.value is not None: return ddp_tree

        if ddp_tree.left is not None and ddp_tree.right is not None:
            self.dpp_to_pddp_tree(ddp_tree.left)
            self.dpp_to_pddp_tree(ddp_tree.right)

        elif ddp_tree.left is not None:
            self.dpp_to_pddp_tree(ddp_tree.left)
            if ddp_tree.left.value is not None:
                ddp_tree.value = ddp_tree.left.value
                ddp_tree.left = None

        else:
            self.dpp_to_pddp_tree(ddp_tree.right)
            if ddp_tree.right.value is not None:
                ddp_tree.value = ddp_tree.right.value
                ddp_tree.right = None

        return ddp_tree

    def dp_to_ddp_tree(self, dp_tree: BinaryEncodingTree) -> BinaryEncodingTree:
        if dp_tree.value is None: # We have no tree_value:
            if dp_tree.left is not None and dp_tree.right is not None:
                self.dp_to_ddp_tree(dp_tree.left)
                self.dp_to_ddp_tree(dp_tree.right)

            if dp_tree.left is not None and dp_tree.right is None:
                self.dp_to_ddp_tree(dp_tree.left)

            if dp_tree.left is None and dp_tree.right is not None:
                self.dp_to_ddp_tree(dp_tree.right)
        else: # We have a tree_value:
            if dp_tree.left is not None and dp_tree.right is not None:
                dp_tree.left.value = dp_tree.value
                dp_tree.value = None
                self.dp_to_ddp_tree(dp_tree.left)
                self.dp_to_ddp_tree(dp_tree.right)

            if dp_tree.left is not None and dp_tree.right is None:
                dp_tree.left.value = dp_tree.value
                dp_tree.value = None
                self.dp_to_ddp_tree(dp_tree.left)

            if dp_tree.left is None and dp_tree.right is not None: # insert new left child
                dp_tree.append_leafs(
                    left=BinaryEncodingTree(
                        value=dp_tree.value
                    )
                )
                dp_tree.value = None

        return dp_tree

    def populate_dp_tree(self, distance: float, sub_tree: BinaryEncodingTree, depth: int = 0, tree_sum: float = 0) -> BinaryEncodingTree:
        alpha_at_depth = alpha[depth]
        next_depth = depth + 1

        if distance == 0 and depth == 0:
            if sub_tree.left is not None:
                return self.populate_dp_tree(distance=distance, sub_tree=sub_tree.left, depth=next_depth, tree_sum=tree_sum)
            else:
                return sub_tree.append_leafs(
                    left=self.populate_dp_tree(distance=distance, sub_tree=BinaryEncodingTree(), depth=next_depth, tree_sum=tree_sum)
                )
        elif distance - tree_sum <= self.err_bound: # Stay
            sub_tree.value = tree_sum
            return sub_tree
        elif distance >= alpha_at_depth + tree_sum - self.err_bound: # Right
            if sub_tree.right is not None:
                return self.populate_dp_tree(distance=distance, sub_tree=sub_tree.right, depth=next_depth, tree_sum=tree_sum + alpha_at_depth)
            else:
                return sub_tree.append_leafs(
                    right=self.populate_dp_tree(distance=distance, sub_tree=BinaryEncodingTree(), depth=next_depth, tree_sum=tree_sum + alpha_at_depth)
                )
        else: # Left
            if sub_tree.left is not None:
                return self.populate_dp_tree(distance=distance, sub_tree=sub_tree.left, depth=next_depth, tree_sum=tree_sum
                )
            else:
                return sub_tree.append_leafs(
                    left=self.populate_dp_tree(distance=distance, sub_tree=BinaryEncodingTree(), depth=next_depth, tree_sum=tree_sum)
                )

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
distance_seq2 = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75]) # 0010011111 encoded
example_trajectory2 = TEDTrajectory(entry_path2, time_flags2, time_seq2, distance_seq2)

trajectories = [example_trajectory, example_trajectory2]

alpha = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024]

if __name__ == '__main__':
    print("Hello TED")
    # print(example_trajectory)
    ted = TEDCompressor(
        k=7,
        num_trajectories=len(trajectories),
        num_entry_paths=len(entry_path)
    )
    ted.compress(trajectories)
