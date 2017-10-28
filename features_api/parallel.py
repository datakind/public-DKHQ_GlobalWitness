import time
import urllib

import gevent
import gevent.monkey
import gevent.threadpool

gevent.monkey.patch_all()


def parallel_map(func, args_tuples, max_threads=8):
    threadpool = gevent.threadpool.ThreadPool(maxsize=max_threads)
    threads = [threadpool.apply_async(func, arg_tuple) for arg_tuple in args_tuples]
    gevent.joinall(threads)
    return [thread.value for thread in threads]
