# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:14:28 2021

@author: baidina
"""
import numpy as np

def to_line(x, y, start=None):    
    x = np.array(x)
    y = np.array(y)
    points = np.c_[x, y]
    data = []
    for k1, node1 in enumerate(points):
        data.append([])
        for k2, node2 in enumerate(points):
            dist = np.linalg.norm(node1-node2)
            data[k1].append(dist)

    
    m = np.argsort(data)[:,1:]
    paths = []
    if start == None:
        for k in range(len(points)):
            order = [k,m[k,0]]
            next_point = order[-1]
            while len(order)<len(points):
                row = m[next_point]
                i = 0
                while row[i] in order:
                    i += 1
                order.append(row[i])
                next_point = order[-1]
            len_path = 0
            for p1, p2 in zip(order[:-1], order[1:]):
                len_path += data[p1][p2]
            paths.append((len_path, order))
        order = min(paths, key=lambda x: x[0])[1]
    else:
        order = [start,m[start,0]]
        next_point = order[-1]
        while len(order)<len(points):
            row = m[next_point]
            i = 0
            while row[i] in order:
                i += 1
            order.append(row[i])
            next_point = order[-1]
        len_path = 0
        for p1, p2 in zip(order[:-1], order[1:]):
            len_path += data[p1][p2]
        #paths.append((len_path, order))
    
    
    return order
    