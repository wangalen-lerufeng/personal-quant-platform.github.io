import time
import unittest

from quant_platform.providers import _RateLimiter


class RateLimiterTests(unittest.TestCase):
    def test_limits_request_rate(self):
        # 600 req/min -> 0.1s interval; 5 acquires should take at least ~0.4s.
        limiter = _RateLimiter(600)
        start = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(elapsed, 0.4)

    def test_is_thread_safe_serialized(self):
        import threading

        limiter = _RateLimiter(1000)
        order = []
        lock = threading.Lock()

        def worker(i):
            limiter.acquire()
            with lock:
                order.append(i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 8 threads, 8 acquires; all should complete.
        self.assertEqual(len(order), 8)


if __name__ == "__main__":
    unittest.main()
