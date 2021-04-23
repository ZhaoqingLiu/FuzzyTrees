"""
@author: Zhaoqing Liu
@email : Zhaoqing.Liu-1@student.uts.edu.au
@date  : 21/4/21 4:41 pm
@desc  :
"""
import multiprocessing
import queue
from multiprocessing.managers import BaseManager


class MyManager(BaseManager):
    pass


if __name__ == '__main__':
    # print("Number of CPU cores of this server: ", multiprocessing.cpu_count())

    task_queue = queue.Queue()
    result_queue = queue.Queue()
