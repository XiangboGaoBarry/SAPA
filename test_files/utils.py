from time import time
import functools
import traceback

class TestWrapper():
    def __init__(self):
        self.test_idx = 0
        
    def timer(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.test_idx += 1
            print(f"TEST {self.test_idx}: Testing [{func.__name__}] ...")
            start = time()
            func(*args, **kwargs)
            print(f"[{func.__name__}] takes {round(time() - start, 4)} seconds")
        return wrapper
    
test_wrapper = TestWrapper()
timer = test_wrapper.timer

# try:
#     timer = test_wrapper.timer
# except:
#     print(traceback.format_exc())
#     print(f"Test {test_wrapper.test_idx} failed, test name: {func.__name__}")
