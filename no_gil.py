"""Testing multithreaded, multiprocess and numba."""
import math
import multiprocessing
import sys
import sysconfig
import threading
import time

import numba
from numba import prange

numba.set_num_threads(12)


def compute_factorial(n):
    """Compute factorial."""
    result = math.factorial(n)
    # print(result)
    return result


# Single-threaded
def single_threaded_compute(n):
    """Single thread computation."""
    for num in n:
        compute_factorial(num)
    print("Single-threaded: Factorial Computed.")


# Multi-threaded
def multi_threaded_compute(n):
    """Multithreaded computation."""
    threads = []
    # Create 5 threads
    for num in n:
        thread = threading.Thread(target=compute_factorial, args=(num,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("Multi-threaded: Factorial Computed.")


# Multi-process
def multi_processing_compute(n):
    """Multiprocess computation."""
    processes = []
    # Create a process for each number in the list
    for num in n:
        process = multiprocessing.Process(target=compute_factorial, args=(num,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("Multi-process: Factorial Computed.")


@numba.jit(fastmath=True, nopython=True)
def compute_numba_factorial(n):
    """Compute factorial numba style."""
    # print(n)
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact


# Numba
@numba.jit(fastmath=True, parallel=True, nopython=True)
def numba_processing_compute(num_list):
    """Process numba compute."""
    # result = 0
    for num in prange(len(num_list)):
        n = num_list[num]
        compute_numba_factorial(n)
        # print(result)


def main():
    """Start computation."""
    # Checking Version
    print(f"Python version: {sys.version}")

    # GIL Status
    status = sysconfig.get_config_var("Py_GIL_DISABLED")
    if status is None:
        print("GIL cannot be disabled")
    if status == 0:
        print("GIL is active")
    if status == 1:
        print("GIL is disabled")

    numlist = [100000, 200000, 300000, 400000, 500000, 100000]

    # numlist = [10, 20]

    # # Single-threaded Execution
    # start = time.time()
    # single_threaded_compute(numlist)
    # end = time.time() - start
    # print(f"Single-threaded time taken: {end:.2f} seconds")

    # # Multi-threaded Execution
    # start = time.time()
    # multi_threaded_compute(numlist)
    # end = time.time() - start
    # print(f"Multi-threaded time taken : {end:.2f} seconds")

    # # Multi-process Execution
    # start = time.time()
    # multi_processing_compute(numlist)
    # end = time.time() - start
    # print(f"Multi-process time taken  : {end:.2f} seconds")

    # Numba-process Execution
    start = time.time()
    numba_processing_compute(numlist)
    end = time.time() - start
    print(f"Numba-process time taken  : {end:.2f} seconds")


if __name__ == "__main__":
    main()
