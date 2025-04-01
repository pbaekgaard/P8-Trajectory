import heapq

class MaxHeap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []  # Stores (-distance, obj)

    def push(self, obj):
        """Pushes an object while ensuring only the smallest distances are kept"""
        # Negate the distance to simulate max heap behavior
        obj["distance"] = -obj["distance"]

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (obj["distance"], obj))
        else:
            # If the new distance is lower than the max in the heap, replace the max element
            if obj["distance"] > self.heap[0][0]:
                heapq.heappushpop(self.heap, (obj["distance"], obj))

    def pop(self):
        """Removes and returns the smallest distance element"""
        if self.heap:
            _, obj = heapq.heappop(self.heap)
            obj["distance"] = -obj["distance"]  # Restore original value
            return obj
        return None

    def peek(self):
        """Returns the largest distance element without removing it"""
        if self.heap:
            return -self.heap[0][0], self.heap[0][1]  # Return max element
        return None

    def get_elements(self):
        """Returns elements sorted by smallest distance"""
        return [obj["trajectory_id"] for _, obj in sorted(self.heap, key=lambda x: x[0], reverse=True)]