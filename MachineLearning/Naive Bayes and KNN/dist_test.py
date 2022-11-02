from unittest import TestCase
import numpy as np
import time
import distances


class Test(TestCase):
    def test_euclidean_for_loop(self):
        """
            Use this to test your for loop implementation
        
            :return:
            """

        # test for 100,000 entries
        array1 = np.random.randint(-100, 100, 100000)
        array2 = np.random.randint(-100, 100, 100000)
        start_time = time.time()
        distance = distances.euclidean_for_loop(array1, array2)
        end_time = time.time()
        print(f' time taken is:', end_time - start_time)

        # test for 1, 000,000 entries
        array3 = np.random.randint(-100, 100, 1000000)
        array4 = np.random.randint(-100, 100, 1000000)
        start_time2 = time.time()
        distance2 = distances.euclidean_for_loop(array3, array4)
        end_time2 = time.time()
        print(f' time taken is:', end_time2 - start_time2)
        assert (abs((distance - np.linalg.norm(array1 - array2))) < 0.1) and (
                abs((distance2 - np.linalg.norm(array3 - array4))) < 0.1)

    def test_euclidean_vectorized(self):
        """
        Use this to test your for loop implementation
        :return:
        """
        # test for 100,000 entries
        array1 = np.random.randint(0, 100, 100000)
        array2 = np.random.randint(0, 100, 100000)
        start_time = time.time()
        distance = distances.euclidean_vectorized(array1, array2)
        end_time = time.time()
        print(f' time taken is:', end_time - start_time)

        # test for 1,000,000 entries
        array3 = np.random.randint(0, 100, 1000000)
        array4 = np.random.randint(0, 100, 1000000)
        start_time2 = time.time()
        distance2 = distances.euclidean_vectorized(array3, array4)
        end_time2 = time.time()
        print(f' time taken is:', end_time2 - start_time2)
        assert (abs((distance - np.linalg.norm(array1 - array2))) < 0.1) and (
                abs((distance2 - np.linalg.norm(array3 - array4))) < 0.1)
