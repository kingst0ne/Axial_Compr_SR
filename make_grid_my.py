# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:36:45 2021

@author: baidina
"""
import sys, os, logging

logging.basicConfig(filename='python_mesh.log',filemode='w',level=logging.DEBUG, 
                    format='%(message)s')#, datefmt='%H:%M:%S')
logger = logging.getLogger('__name__')
logger.info('Log File Created\nPyMesher started working\n')

import time
import msh
import meshio
import numpy as np
from scipy.interpolate import splprep, splev
import scipy.spatial.distance as distance
from collections import Counter, namedtuple

from toLine import to_line

FILE_PATH = os.getcwd()
start_time = time.time()

class Primitiv:
    def __init__(self, points, ID, ptype):
        self.points = points
        self.index = len(points)
        self.ID = ID
        self.X = np.array(points)[:,0]
        self.Y = np.array(points)[:,1]
        self.type = ptype
        self.set_points = []
        self.set_points_id = []
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.points[self.index]

def main(filename=None):  
    el_len = 0
    min_el_len = 0
    dist_fun_koef = 0
    set_contact = 0
    if len(sys.argv) < 2:
        el_len = 1
        min_el_len = 1
        dist_fun_koef = 0.2
        if filename == None:
            filename = [f for f in os.listdir(FILE_PATH) if f.endswith('test.dat')][0]
        
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        if len(filename.split('.')) < 2:
            raise IOError('Неверное имя файла!')
        el_len = 3
        min_el_len = el_len
        dist_fun_koef = 0.2
    
    elif len(sys.argv) > 2:
        filename = sys.argv[1]
        if len(filename.split('.')) < 2:
            raise IOError('Неверное имя файла!')
        for arg in sys.argv[2:]:
            spl_arg = arg.split('=')
            if 'elsize' in spl_arg[0]:
                el_len = float(spl_arg[1])
            elif 'refinesize' in spl_arg[0]:
                min_el_len = float(spl_arg[1])
            elif 'coeff' in spl_arg[0]:
                dist_fun_koef = float(spl_arg[1])
            elif 'contact' in spl_arg[0]:
                set_contact = int(spl_arg[1])
            
                
    if el_len == 0: el_len = 3
    if min_el_len == 0: min_el_len = el_len
    if dist_fun_koef == 0: dist_fun_koef = 0.2
        
    
        
    
    geo_cont = None
    contact_nodes = []
    
    if set_contact == 1:
        if os.path.exists(FILE_PATH + "/points.dat"):
            with open("points.dat", "r", encoding='utf-8', newline='') as file:
                for line in file:
                    try:
                        contact_nodes.append(list(map(float, line.split(';'))))
                    except:
                        pass
            contact_nodes = np.array(contact_nodes)
            order = to_line(contact_nodes[:,0], contact_nodes[:,1])
            contact_nodes = contact_nodes[order]
            geo_cont = msh.Path(contact_nodes)
        else:
            print("points.dat not exists")
     
    
    Point = namedtuple('Point', ['x', 'y'])
    primitives = []
    primitivesInd = []
    points_n = []
    key = ''
    ID = 0
    is_curve = False
    is_shred = False
    
    with open(filename, "r") as file:
        k = 0
        for line in file:
            sp = line.strip().split(",")
            if sp[2][2:] != key or key == 'Line':
                key = sp[2][2:]
                ID += 1
                primitives.append({key: []})
                primitivesInd.append({key: []})
            points_n.append((float(sp[0]), float(sp[1])))
            primitives[ID-1][key].append((float(sp[0]), float(sp[1])))
            primitivesInd[ID-1][key].append((float(sp[0]), float(sp[1]), k))
            k += 1
            if 'Curve' in line and is_curve == False: is_curve = True
    
    if is_curve and min_el_len < el_len: is_shred = True
    
    ar_points = np.array(points_n)
    
    bound_min, bound_max = (np.array([min(ar_points[:,0]), min(ar_points[:,1])]),
                            np.array([max(ar_points[:,0]), max(ar_points[:,1])]))
    
    mazes = []
    count_x = list(Counter(ar_points[:,1]).items())
    targs_x = np.array(sorted([(k, ar_points[:,1].tolist().count(k)) for k,v in count_x if v>5],
                   key = lambda x: x[1]))
    
    count_y = list(Counter(ar_points[:,0]).items())
    targs_y = np.array(sorted([(k, ar_points[:,0].tolist().count(k)) for k,v in count_y if v>5],
                   key = lambda x: x[1]))
    tol = 0.005
    if len(targs_x) > 0:
        mazes = ar_points[ar_points[:,1] < max(targs_x[:,0]) + tol]
        mazes = mazes[mazes[:,1] > min(targs_x[:,0]) - tol]
        
    if len(targs_y) > 0:
        mazes = ar_points[ar_points[:,0] < max(targs_y[:,0]) + tol]
        mazes = mazes[mazes[:,0] > min(targs_y[:,0]) - tol]
    
    if len(mazes) > 0:
        Y = np.diagonal(distance.cdist(mazes[:-1], mazes[1:])).tolist()
        av_maze = np.average(Y)
        ind = Y.index(max(Y))
        av1 = np.average(Y[:ind])
        av2 = np.average(Y[ind+1:])
        av_b = np.average(bound_max - bound_min)
        mz_p1 = []
        mz_p2 = []
        mazes_all = []
        if max(Y) > av_b*0.1:  #abs(av1 - av_b)/av1 > 2 and abs(av2 - av_b)/av2 > 2 and 
            maze1 = mazes[:ind]
            maze2 = mazes[ind+1:]
            min_x1 = min(maze1, key = lambda x: x[0])[0]
            min_y1 = min(maze1, key = lambda x: x[1])[1]
            max_x1 = max(maze1, key = lambda x: x[0])[0]
            max_y1 = max(maze1, key = lambda x: x[1])[1]
            # for mzp in maze1:
            #     mnode = Point(*mzp)
            #     if case == 2:
            #         if abs(mnode.x - min_x1) < 0.0005 or abs(mnode.x - max_x1) < 0.0005:
            #             if mzp not in mz_p1: mz_p1.append(mzp)
            #     elif case == 1:
            #         if abs(mnode.y - min_y1) < 0.0005 or abs(mnode.y - max_y1) < 0.0005:
            #             if mzp not in mz_p1: mz_p1.append(mzp)
                
            for prim in primitives:
                if 'Line' in prim.keys():
                    mzp = list(prim.values())[0][0]
                    mnode = Point(*mzp)
                    if min_x1 - tol < mnode.x < max_x1 + tol and min_y1 - tol < mnode.y < max_y1 + tol:
                        if mzp not in mz_p1: mz_p1.append(mzp)
                if 'Curve' in prim.keys():
                    mzp = list(prim.values())[0][0]
                    mnode = Point(*mzp)
                    if min_x1 - tol < mnode.x < max_x1 + tol and min_y1 - tol < mnode.y < max_y1 + tol:
                        if mzp not in mz_p1: mz_p1.append(mzp)
                    mzp1 = list(prim.values())[0][-1]
                    mnode1 = Point(*mzp1)
                    if min_x1 - tol < mnode1.x < max_x1 + tol and min_y1 - tol < mnode1.y < max_y1 + tol:
                        if mzp1 not in mz_p1: mz_p1.append(mzp1)
                        
                        
                        
            min_x2 = min(maze2, key = lambda x: x[0])[0]
            min_y2 = min(maze2, key = lambda x: x[1])[1]
            max_x2 = max(maze2, key = lambda x: x[0])[0]
            max_y2 = max(maze2, key = lambda x: x[1])[1]
            # for mzp in maze2:
            #     mnode = Point(*mzp)
            #     if case == 2:
            #         if abs(mnode.x - min_x2) < 0.0005 or abs(mnode.x - max_x2) < 0.0005:
            #             if mzp not in mz_p2: mz_p2.append(mzp)
            #     elif case == 1:
            #         if abs(mnode.y - min_y2) < 0.0005 or abs(mnode.y - max_y2) < 0.0005:
            #             if mzp not in mz_p2: mz_p2.append(mzp)
                        
            for prim in primitives:
                if 'Line' in prim.keys():
                    mzp = list(prim.values())[0][0]
                    mnode = Point(*mzp)
                    if min_x2 - tol < mnode.x < max_x2 + tol and min_y2 - tol < mnode.y < max_y2 + tol:
                        if mzp not in mz_p2: mz_p2.append(mzp)
                if 'Curve' in prim.keys():
                    mzp = list(prim.values())[0][0]
                    mnode = Point(*mzp)
                    if min_x2 - tol < mnode.x < max_x2 + tol and min_y2 - tol < mnode.y < max_y2 + tol:
                        if mzp not in mz_p2: mz_p2.append(mzp)
                    mzp1 = list(prim.values())[0][-1]
                    mnode1 = Point(*mzp)
                    if min_x2 - tol < mnode1.x < max_x2 + tol and min_y2 - tol < mnode1.y < max_y2 + tol:
                        if mzp not in mz_p2: mz_p2.append(mzp)
            if len(mz_p1) > 1: mazes_all.append(mz_p1)
            if len(mz_p2) > 1: mazes_all.append(mz_p2)
                        
            #mazes_all = [maze1, maze2]
        else:
            maze1 = mazes
            min_x1 = min(maze1, key=lambda x: x[0])[0]
            min_y1 = min(maze1, key=lambda x: x[1])[1]
            max_x1 = max(maze1, key=lambda x: x[0])[0]
            max_y1 = max(maze1, key=lambda x: x[1])[1]
            # for mzp in points_n:
            #     mnode = Point(*mzp)
            #     if case == 2:
            #         if abs(mnode.x - min_x1) < 0.0005 or abs(mnode.x - max_x1) < 0.0005:
            #             if mzp not in mz_p1: mz_p1.append(mzp)
            #     elif case == 1:
            #         if abs(mnode.y - min_y1) < 0.0005 or abs(mnode.y - max_y1) < 0.0005:
            #             if mzp not in mz_p1: mz_p1.append(mzp)
            
            for prim in primitives:
                if 'Line' in prim.keys():
                    mzp = list(prim.values())[0][0]
                    mnode = Point(*mzp)
                    if min_x1 - tol < mnode.x < max_x1 + tol and min_y1 - tol < mnode.y < max_y1 + tol:
                        if mzp not in mz_p1: mz_p1.append(mzp)
                if 'Curve' in prim.keys():
                    mzp = list(prim.values())[0][0]
                    mnode = Point(*mzp)
                    if min_x1 - tol < mnode.x < max_x1 + tol and min_y1 - tol < mnode.y < max_y1 + tol:
                        if mzp not in mz_p1: mz_p1.append(mzp)
                    mzp1 = list(prim.values())[0][-1]
                    mnode1 = Point(*mzp1)
                    if min_x1 - tol < mnode1.x < max_x1 + tol and min_y1 - tol < mnode1.y < max_y1 + tol:
                        if mzp1 not in mz_p1: mz_p1.append(mzp1)
            if len(mz_p1) > 1: mazes_all.append(mz_p1)
            #mazes_all = [maze1]
    
    
    for n, val in enumerate(primitives):
        X = list(val.values())[0]
        if 'Curve' in list(val.keys())[0]:
            if len(X) < 3:
                primitives[n] = {'line': X}
            else:
                Y = np.diagonal(distance.cdist(X[:-1], X[1:])).tolist()
                ind = Y.index(max(Y))
                arc1 = np.average(Y[:ind])
                arc2 = np.average(Y[ind+1:])
                if abs(arc1 - max(Y))/arc1 > 2 and abs(arc2 - max(Y))/arc2 > 2:
                    primitives.remove(primitives[n])
                    primitives.insert(n, {'Curve': X[:ind]})
                    primitives.insert(n+1, {'Line': [X[ind]]})
                    primitives.insert(n+2, {'Curve': X[ind+1:]})
    
    
    #tree = KDTree(points_n)
    
    dict_sets2 = []
    
    for n, prim in enumerate(primitivesInd):
        if 'Line' in prim.keys():
            nn = list(prim.values())[0][0][-1]
            dict_sets2.append([nn-1, nn, 'Line'])
        if 'Curve' in prim.keys():
            nn = np.array(list(prim.values())[0], dtype=np.int32)[:,-1]
            if nn[0] != 0:
                dict_sets2.append(list(np.arange(nn[0] - 1, nn[-1]+1)) + ['Curve'])
            else:
                dict_sets2.append(list(np.arange(nn[0], nn[-1]+1)) + ['Curve'])
    
    new_points = []
    paths = []
    ID = 0
    primitives1 = {}
    id_m1 = 0
    id_m2 = 0
    for val in dict_sets2:
        if 'Line' in val:
            ID += 1
            primitives1[ID] = Primitiv(ar_points[val[:-1]], ID, 'Line')
            nod1 = Point(*points_n[val[0]])
            nod2 = Point(*points_n[val[1]])
            nx = np.linspace(nod1.x, nod2.x, 1000)
            ny = np.linspace(nod1.y, nod2.y, 1000)
            if len(mazes) > 0:
                if all([min_x1 <= points_n[_][0] <= max_x1 and
                        min_y1 <= points_n[_][1] <= max_y1 for _ in val[:-1]]):
                    if id_m1 ==0:
                        id_m1 = ID
                        if el_len == min_el_len:
                            pass
                        else:
                            new_points.extend(mz_p1)
                            paths.append(msh.Path(mz_p1))
                    continue
                
                elif len(mazes_all) > 1 and all([min_x2 <= points_n[_][0] <= max_x2 and
                        min_y2 <= points_n[_][1] <= max_y2 for _ in val[:-1]]):
                    if id_m2 ==0:
                        id_m2 = ID
                        if el_len == min_el_len:
                            pass
                        else:
                            new_points.extend(mz_p2)
                            paths.append(msh.Path(mz_p2))
                    continue
            
            in_contact = False
            if len(contact_nodes) > 0:
                contact_pp = []
                dist_from_start = []
                max_nx = max([nod1.x,nod2.x])
                min_nx = min([nod1.x,nod2.x])
                max_ny = max([nod1.y,nod2.y])
                min_ny = min([nod1.y,nod2.y])
                for _node in contact_nodes:
                    if min_nx-0.005 <= _node[0] <= max_nx+0.005 and min_ny-0.005 <= _node[1] <= max_ny+0.005: # and tuple(_node) not in new_points:
                        #new_points.append(tuple(_node))
                        contact_pp.append(tuple(_node))
                        in_contact = True
                if len(contact_pp) > 1:
                    dist_from_start = [np.linalg.norm(ar_points[val[0]] - _n) for _n in contact_pp]
                    contact_pp = np.array(contact_pp)
                    for _node in contact_pp[np.argsort(dist_from_start)]:
                        new_points.append(tuple(_node))
                        
                el_len = np.max(np.diagonal(distance.cdist(contact_nodes[:-1], contact_nodes[1:])))
                
            if not is_shred and not in_contact:
                if (nod1.x, nod1.y) not in new_points:
                    new_points.append((nod1.x, nod1.y))
                if (nod2.x, nod2.y) not in new_points:
                    new_points.append((nod2.x, nod2.y)) 
    
            elif not in_contact:
                len_line = distance.cdist([ar_points[val[0]]], [ar_points[val[1]]])
                if len_line > 0.1*el_len:
                    Nmax = int(len_line[0][0]//el_len)
                    if Nmax > 0:
                        chop_x = [nx[_:_ + int(1000/Nmax)][-1] for _ in range(0, len(nx), int(1000/Nmax+1))]
                        chop_y = [ny[_:_ + int(1000/Nmax)][-1] for _ in range(0, len(ny), int(1000/Nmax+1))]
                        if points_n[val[0]] not in list(zip(chop_x, chop_y)):
                            chop_x.insert(0, points_n[val[0]][0])
                            chop_y.insert(0, points_n[val[0]][1])
    
                        if points_n[val[1]] not in list(zip(chop_x, chop_y)):
                            chop_x.insert(-1, points_n[val[1]][0])
                            chop_y.insert(-1, points_n[val[1]][1])
    
                        new_points.extend(list(zip(chop_x, chop_y)))
                    else:
                        new_points.extend([(nod1.x, nod1.y), (nod2.x, nod2.y)])
    
        if 'Curve' in val:
            ID += 1
            primitives1[ID] = Primitiv(ar_points[val[:-1]], ID, 'Curve')
            if len(mazes) > 0:
                if all([min_x1 <= points_n[_][0] <= max_x1 and
                        min_y1 <= points_n[_][1] <= max_y1 for _ in val[:-1]]):
                    pass
                elif len(mazes_all) > 1 and all([min_x2 <= points_n[_][0] <= max_x2 and
                        min_y2 <= points_n[_][1] <= max_y2 for _ in val[:-1]]):
                    pass
                else:
                    X = ar_points[val[:-1]]
                    X1 = X.tolist()
                    for xp in X.tolist():
                        if xp in np.array(mz_p1) or xp in np.array(mz_p2): X1.remove(xp)
                    X = np.array(X1)
                    Y = np.diagonal(distance.cdist(X[:-1], X[1:]))
                    Nmax = sum(Y)//min_el_len
                    if Nmax <= 1: Nmax = 3
                    tck, u = splprep(X.T, u=None, s=0.0, per=0)
                    u_new = np.linspace(u.min(), u.max(), int(Nmax))
                    x_new, y_new = splev(u_new, tck, der=0)
        
                    xy = list(zip(x_new, y_new))
                    new_points.extend(xy)
        
                    paths.append(msh.Path(X[1:-1]))
            else:
                X = ar_points[val[:-1]]
                Y = np.diagonal(distance.cdist(X[:-1], X[1:]))
                Nmax = sum(Y)//min_el_len
                if Nmax <= 1: Nmax = 3
                tck, u = splprep(X.T, u=None, s=0.0, per=0)
                u_new = np.linspace(u.min(), u.max(), int(Nmax))
                x_new, y_new = splev(u_new, tck, der=0)
                xy = list(zip(x_new, y_new))
                new_points.extend(xy)
                paths.append(msh.Path(X[1:-1]))
                    
    for np1, np2 in zip(new_points[:-1], new_points[1:]):
        if np.allclose(np1, np2, rtol = 1e-4*min([el_len, min_el_len])):
            new_points.remove(np2)
    
    if np.allclose(new_points[0], new_points[-1], rtol = 1e-4*min([el_len, min_el_len])):
        new_points = new_points[:-1]
        
    for p in new_points:
        if new_points.count(p) > 1:
            new_points.remove(p)
    
    if len(paths) > 1:
        P = msh.Union(paths)
    elif len(paths) == 0:
        P = 0
    else:
        P = paths[0]
    
    
    geo = msh.Polygon(new_points[:])
    if not is_shred:
        X, cells, areas = msh.generate(geo, el_len, geo_cont=geo_cont)
        
    elif P == 0:
        min_el_len = el_len
        X, cells, areas = msh.generate(geo, el_len, geo_cont=geo_cont)
        #raise IOError('Не найдены участки для замельчения')
        
    else:
        X, cells, areas = msh.generate(
            geo, lambda x: x,
            min_edge_size = min_el_len,
            max_edge_size = el_len,
            dist_fun_koef = dist_fun_koef,
            geo_min = P,
            tol=1.0e-10,
            geo_cont=geo_cont
        )
        
    #msh.show(X, cells, geo)
    mid = np.sum(X[cells], axis=1)/3
    masses = np.c_[mid, areas]
    CM = np.average(masses[:,:2], axis=0, weights=masses[:,2])
    
    # import matplotlib.pyplot as plt
    # plt.plot(CM[0], CM[1], '.', c='r')
    
    
    trash = []
    for i in range(len(X)):
        if not i in cells:
            print(i)
            trash.append(i)
            X = X[~(X == X[i]).all(axis=1)]
    
    for i in sorted(trash):
        cells = np.where(cells > i, cells - 1, cells)
        
    qual = [msh.quality(X[i]) for i in cells]
    boundPoints = msh.boundary_points(X, cells)
    inds = msh.boundary_points_ids(X, cells)
    
    tol = 1e-4
    for fp, ind in zip(boundPoints, inds):
        for prim in primitives1:
            val = primitives1[prim]
            if min(val.X)-tol <= fp[0] <= max(val.X)+tol and min(val.Y)-tol <= fp[1] <= max(val.Y)+tol:
                val.set_points.append(fp)
                val.set_points_id.append(ind)
    
    points_sets =[(primitives1[prim].ID, primitives1[prim].set_points_id) for prim in primitives1]
    
    meshio.write_points_cells(f"{filename.split('.')[0]}_mesh.inp", X, {"triangle2": cells}, point_sets=dict(points_sets))
    logger.info('\n')
    logger.info('Successful meshing!')
    logger.info(f'Working time: {"%.3f" %(time.time() - start_time)}')
    logger.info(f'Num of nodes: {len(X)}')
    logger.info(f'Num of elements: {len(cells)}')
    
    return CM

if __name__ == '__main__':
    try:
        CM = main()
    except Exception as err:
        print(err)
        logger.error(err, exc_info=True)



