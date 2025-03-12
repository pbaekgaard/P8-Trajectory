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

    def serialize(self) -> np.ndarray:
        result = []
        self._serialize_helper(self, result)
        return np.array(result)

    def _serialize_helper(self, node, result):
        if node is None:
            return
        else:
            result.append(node.value)
            self._serialize_helper(node.left, result)
            self._serialize_helper(node.right, result)


def deserialize_pddp_tree(serialized_data: List) -> BinaryEncodingTree:
    def helper(index):
        if index >= len(serialized_data):
            return None, index

        value = serialized_data[index]
        index += 1

        node = BinaryEncodingTree(value)
        if node.value is not None:
            return node, index

        if node.value is None:
            node.left, index = helper(index)
            node.right, index = helper(index)

        return node, index

    tree, _ = helper(0)
    return tree

class TEDTrajectory(object):
    def __init__(self, entry_path: np.ndarray, time_flags: np.ndarray, time_seq: np.ndarray, distance_seq: np.ndarray):
        super().__init__()
        self.entry_path = entry_path
        self.time_flags = time_flags
        self.time_seq = time_seq
        self.distance_seq = distance_seq

# Also for packing using np.packbits
class TEDCompressed(object):
    def __init__(self, entry_path_primes: np.ndarray = None, A: np.ndarray = None, B: np.ndarray = None, time_flags: np.ndarray = None,
                 time_seqs: np.ndarray = None, pddp_trees: np.ndarray = None, encoded_distances: np.ndarray = None, value: bool = False):
        super().__init__()
        if value: return

        self.shape_A = A.shape[1]
        self.shape_B = B.shape[0]
        self.shape_time_flags = time_flags.shape[1]
        self.shape_entry_path_primes = entry_path_primes.shape[1]

        self.original_lengths_encoded_distances = np.array([len(row) for row in encoded_distances], dtype=np.uint16)

        packed_arrays = []
        for row in encoded_distances:
            #pad_size = (8 - len(row) % 8) % 8  # Calculate padding
            #padded_row = np.pad(row, (0, pad_size), mode='constant')  # Pad with zeros
            packed_arrays.append(np.packbits(row))


        # Convert to object dtype NumPy array
        self.encoded_distances = np.array(packed_arrays, dtype=object)

        self.entry_path_primes = np.packbits(entry_path_primes, axis=1)
        self.A = np.packbits(A, axis=1)
        self.B = np.packbits(B)
        self.time_seqs = time_seqs
        self.pddp_trees = pddp_trees
        self.time_flags = np.packbits(time_flags, axis=1)
        # TODO: Use this when we implement query processing.
        # unpacked_A = np.unpackbits(self.A, axis=1, count=self.shape_A)
        # unpacked_B = np.unpackbits(self.B, count=self.shape_B)
        # unpacked_time_flags = np.unpackbits(self.time_flags, axis=1, count=self.shape_time_flags)
        # unpacked_entry_paths = np.unpackbits(self.entry_path_primes, axis=1, count=self.shape_entry_path_primes)

        #unpacked_arrays = []
        #for packed, orig_len in zip(self.encoded_distances, self.original_lengths_encoded_distances):
            #unpacked = np.unpackbits(packed, count=orig_len)  # Unpack and remove padding
            #unpacked_arrays.append(unpacked)

        # Convert back to a variable-length NumPy array (dtype=object)
        #unpacked_encoded_distances = np.array(unpacked_arrays, dtype=object)

    def save_to_file(self, filename):
        np.savez_compressed(filename,
                            shape_A=self.shape_A,
                            shape_B=self.shape_B,
                            shape_time_flags=self.shape_time_flags,
                            shape_entry_path_primes=self.shape_entry_path_primes,
                            entry_path_primes=self.entry_path_primes,
                            A=self.A,
                            B=self.B,
                            time_flags=self.time_flags,
                            time_seqs=self.time_seqs,
                            pddp_trees=self.pddp_trees,
                            encoded_distances=self.encoded_distances,
                            original_lengths_encoded_distances=self.original_lengths_encoded_distances
        )

    @staticmethod
    def load_ted_compressed(filename):
        data = np.load(filename, allow_pickle=True)
        ted_compressed = TEDCompressed(value=True)
        ted_compressed.shape_A = data["shape_A"]
        ted_compressed.shape_B = data["shape_B"]
        ted_compressed.shape_time_flags = data["shape_time_flags"]
        ted_compressed.shape_entry_path_primes = data["shape_entry_path_primes"]
        ted_compressed.entry_path_primes = data["entry_path_primes"]
        ted_compressed.A = data["A"]
        ted_compressed.B = data["B"]
        ted_compressed.time_flags = data["time_flags"]
        ted_compressed.time_seqs = data["time_seqs"]
        ted_compressed.pddp_trees = data["pddp_trees"]
        ted_compressed.encoded_distances = data["encoded_distances"]
        ted_compressed.original_lengths_encoded_distances = data["original_lengths_encoded_distances"]
        return ted_compressed


class TEDCompressor(object):
    def __init__(self, k: int, num_trajectories, num_entry_paths):
        super().__init__()
        self.k = math.floor(math.log2(k) + 1)
        self.shape = (num_trajectories, num_entry_paths)
        self.err_bound = 0.02 # TODO: Calculate dis, no magic numbers

    def compress(self, ted_trajectories: List[TEDTrajectory]) -> TEDCompressed:
        M = np.empty(self.shape, dtype=np.uint8)
        entry_path_primes = np.empty((self.shape[0], self.shape[1] * (self.k - 1)), dtype=np.uint8)
        pddp_tree_vector = np.empty((self.shape[0]), dtype=object)  # one pr trajectory
        encoded_distances_matrix = np.empty((self.shape[0]), dtype=object)  # one pr trajectory | DTYPE MUST BE OBJECT!!! str not working in numpy.
        T_prime_matrix = np.empty((self.shape[0]), dtype=object)  # GIGA MATRICE
        flags_matrix = np.empty(self.shape, dtype=np.uint8)

        for index, ted_trajectory in enumerate(ted_trajectories):
            # compression of entry_paths:
            M[index], entry_path_primes[index] = self.compress_entry_path(ted_trajectory.entry_path)

            # compression of distance seq:

            pddp_tree, encoded_distances_matrix[index] = self.compress_distance_seq(ted_trajectory.distance_seq)
            pddp_tree_vector[index] = pddp_tree.serialize()

            # compression of time_seq
            T_prime_matrix[index] = self.compress_time_seq(ted_trajectory.time_seq)

            # time flags are packed in compressor
            flags_matrix[index] = ted_trajectory.time_flags

        A, B = self.compress_M(M) # GIGA MATRICE

        # TODO: Compress all binary representations with np.packbits
        return TEDCompressed(
            entry_path_primes=entry_path_primes,
            A=A,
            B=B,
            time_flags=flags_matrix,
            time_seqs=T_prime_matrix,
            pddp_trees=pddp_tree_vector,
            encoded_distances=encoded_distances_matrix
        )

    def compress_time_seq(self, time_seq: np.ndarray) -> np.ndarray:
        time_seq = time_seq[~np.isnan(time_seq)].astype(np.uint32)
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

    def compress_M(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        columns_with_one = np.any(M == 1, axis=0)
        B = np.zeros(M.shape[1], dtype=np.uint8)
        A = np.empty((M.shape[0], len(np.where(columns_with_one)[0])), dtype=np.uint8)


        index_A = 0
        A = A.T

        for (idx, col) in enumerate(M.T):
            if np.any(col == 1):
                B[idx] = 1
                A[index_A] = col
                index_A += 1

        return A, B

    def compress_entry_path(self, entry_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        entry_path_prime = self.int_to_binary(entry_path)
        entry_path_str = ""
        M_row = np.empty(np.shape(entry_path_prime), dtype=np.uint8)
        index = 0

        for entry in np.nditer(entry_path_prime):
            M_row[index] = str(entry)[0]
            entry_path_str += str(entry)[1:]
            index += 1

        entry_path_uint8_array = np.array(list(entry_path_str), dtype=np.uint8)
        return M_row, entry_path_uint8_array

    def compress_distance_seq(self, distance_seq: np.ndarray) -> Tuple[BinaryEncodingTree, np.ndarray]:
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

        encoded_distances_uint8_array = np.array(list(encoded_string), dtype=np.uint8)
        return pddp_tree, encoded_distances_uint8_array


    def encode_pddp_tree(self, distance: float, pddp_tree: BinaryEncodingTree, encoded_string: str = "", tree_sum: float = 0, depth: int = 1) -> str:
        new_depth = depth + 1
        alpha_at_depth = 1 / (2 ** depth)

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

        elif ddp_tree.right is not None:
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

    def populate_dp_tree(self, distance: float, sub_tree: BinaryEncodingTree, depth: int = 1, tree_sum: float = 0) -> BinaryEncodingTree:
        next_depth = depth + 1
        alpha_at_depth = 1 / (2 ** depth)

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
time_flags = np.array([1,0,1,1,1,1])
time_seq = np.array([0, np.nan, 90, 180, 270, 361])
distance_seq = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75])
example_trajectory = TEDTrajectory(entry_path, time_flags, time_seq, distance_seq)

entry_path2 = np.array([1, 1, 3, 1, 2, 4])
time_flags2 = np.array([1,0,1,1,1,1])
time_seq2 = np.array([0, np.nan, 90, 180, 270, 361])
distance_seq2 = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75])
example_trajectory2 = TEDTrajectory(entry_path2, time_flags2, time_seq2, distance_seq2)

entry_path3 = np.array([4, 0, 1, 2, 1, 2])
time_flags3 = np.array([1,0,1,1,1,1])
time_seq3 = np.array([0, np.nan, 90, 180, 270, 361])
distance_seq3 = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75])
example_trajectory3 = TEDTrajectory(entry_path3, time_flags3, time_seq3, distance_seq3)

#entry_path2 = np.array([3, 4, 2, 1, 0, 6])
#time_flags2 = np.array([1, 0, 1, 1, 1, 1])
#time_seq2 = np.array([0, np.nan, 90, 180, 270, 361])
#distance_seq2 = np.array([0, np.nan, 0.5, 0.375, 0.75, 0.75]) # 0010011111 encoded
#example_trajectory2 = TEDTrajectory(entry_path2, time_flags2, time_seq2, distance_seq2)


trajectories = [example_trajectory, example_trajectory2, example_trajectory3]

for index in range(0, 999997):
    entry_path4 = np.random.randint(0, 6, size=6)
    time_flags4 = np.random.randint(0, 1, size=6)
    time_seq4 = np.sort(np.where(time_flags4 == 1, np.random.randint(0, 1000, size=6), np.nan))
    distance_seq4 = np.where(time_flags4 == 1, np.random.uniform(0, 1, size=6), np.nan)
    example_trajectory4 = TEDTrajectory(entry_path4, time_flags4, time_seq4, distance_seq4)
    trajectories.append(example_trajectory4)

if __name__ == '__main__':
    print("Hello TED")
    # print(example_trajectory)
    ted = TEDCompressor(
        k=7,
        num_trajectories=len(trajectories),
        num_entry_paths=len(entry_path)
    )
    #print(trajectories)
    compressed_trajectories = ted.compress(trajectories)
    compressed_trajectories.save_to_file("data.npz")
    data = TEDCompressed.load_ted_compressed("data.npz")
    print("")
