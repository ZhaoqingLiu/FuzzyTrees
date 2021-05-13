# _*_coding:utf-8_*_
"""
@author: Zhaoqing Liu
@email: Zhaoqing.Liu-1@student.uts.edu.au
@date: 08/05/2021 9:35 pm
@desc: 
"""
import time


def get_timestamp_str():
    return str(int(round(time.time() * 1000)))


def get_today_str():
    return time.strftime("%Y-%m-%d", time.localtime(time.time()))


def get_now_str(tim_str):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(tim_str)/1000))


if __name__ == '__main__':
    now_str = get_timestamp_str()
    print(now_str)
    print(get_now_str(now_str))
    print(get_today_str())
