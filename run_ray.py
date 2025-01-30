import ray
import numpy as np


def ray_task_example():
    # Define a task that sums the values in a matrix.
    @ray.remote
    def sum_matrix(matrix):
        return np.sum(matrix)

    # Call the task with a literal argument value.
    print(ray.get(sum_matrix.remote(np.ones((100, 100)))))
    # -> 10000.0

    # Put a large array into the object store.
    matrix_ref = ray.put(np.ones((1000, 1000)))

    # Call the task with the object reference as an argument.
    print(ray.get(sum_matrix.remote(matrix_ref)))
    # -> 1000000.0



def ray_actor_example():

    # Define the Counter actor.
    @ray.remote
    class Counter:
        def __init__(self):
            self.i = 0

        def get(self):
            return self.i

        def incr(self, value):
            self.i += value

    # Create a Counter actor.
    c = Counter.remote()

    # Submit calls to the actor. These calls run asynchronously but in
    # submission order on the remote actor process.
    for _ in range(10):
        c.incr.remote(1)

    # Retrieve final actor state.
    print(ray.get(c.get.remote()))
    # -> 10

if __name__ == "__main__":
    
    print("=" * 100)
    print("ray init")
    ray.init()
    
    print("=" * 100)
    ray_task_example()
    
    print("=" * 100)
    ray_actor_example()

    print("=" * 100)
    print("ray shutdown")
    ray.shutdown()
    
    print("=" * 100)



