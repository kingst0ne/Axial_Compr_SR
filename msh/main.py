# import pkgutil
# search_path = ['.'] # Используйте None, чтобы увидеть все модули, импортируемые из sys.path
# all_modules = [x[1] for x in pkgutil.iter_modules(path=search_path)]
# print(all_modules)

import logging
import meshplex
import numpy as np
import scipy.spatial
import scipy.spatial.distance as distance
#import matplotlib.pyplot as plt


from typing import Callable, Union

logger = logging.getLogger('__name__')
def show(pts, cells, geo, title=None, full_screen=True):
    import matplotlib.pyplot as plt
    eps = 1.0e-10
    is_inside = geo.dist(pts.T) < eps
    # plt.plot(pts[is_inside, 0], pts[is_inside, 1], ".")
    # plt.plot(pts[~is_inside, 0], pts[~is_inside, 1], ".", color="r")
    plt.triplot(pts[:, 0], pts[:, 1], cells)
    plt.axis("square")

    #show cells indices
    # for idx, barycenter in enumerate(np.sum(pts[cells], axis=1) / 3):
    #     plt.plot(*barycenter, ".")
    #     plt.text(
    #         *barycenter, idx, horizontalalignment="center", verticalalignment="center"
    #     )

    #show node indices
    # for idx, pt in enumerate(pts):
    #     plt.text(
    #         *pt, idx, horizontalalignment="center", verticalalignment="center"
    #     )

    if full_screen:
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.showMaximized()
        except AttributeError:
            pass

    if title is not None:
        plt.title(title)

    try:
        geo.show(level_set=False)
    except AttributeError:
        pass
    pass

def center_masses(pts, cells, areas):
    mid = np.sum(pts[cells], axis=1)/3
    masses = np.c_[mid, areas]
    CM = np.average(masses[:,:2], axis=0, weights=masses[:,2])
    return CM

def quality(cell):
    a, b, c = np.diagonal(distance.cdist(cell[np.array([0,1,2])], 
                                         cell[np.array([1,2,0])]))
    return (b + c - a)*(c + a - b)*(a + b - c)/(a*b*c)

def boundary_points(pts, cells):
    mesh = meshplex.MeshTri(pts, cells)
    return pts[mesh.is_boundary_point]

def boundary_points_ids(pts, cells):
    mesh = meshplex.MeshTri(pts, cells)
    pts_inds = np.array(list(range(len(pts))))
    return pts_inds[mesh.is_boundary_point]

def _create_cells(pts, geo):
    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(pts)
    cells = tri.simplices.copy()

    # kick out all cells whose barycenter is not in the geometry
    bc = np.sum(pts[cells], axis=1) / 3.0
    cells = cells[geo.dist(bc.T) < 0.0]

    # kick out all cells whose barycenter or edge midpoints are not in the geometry
    # btol = 1.0e-3
    # bc = np.sum(pts[cells], axis=1) / 3.0
    # barycenter_inside = geo.dist(bc.T) < btol
    # # Remove cells which are (partly) outside of the domain. Check at the midpoint of
    # # all edges.
    # mid0 = (pts[cells[:, 1]] + pts[cells[:, 2]]) / 2
    # mid1 = (pts[cells[:, 2]] + pts[cells[:, 0]]) / 2
    # mid2 = (pts[cells[:, 0]] + pts[cells[:, 1]]) / 2
    # edge_midpoints_inside = (
    #     (geo.dist(mid0.T) < btol)
    #     & (geo.dist(mid1.T) < btol)
    #     & (geo.dist(mid2.T) < btol)
    # )
    # cells = cells[barycenter_inside & edge_midpoints_inside]
    return cells

def _recell_and_boundary_step(mesh, geo, flip_tol):
    # We could do a _create_cells() here, but inverted boundary cell removal plus Lawson
    # flips produce the same result and are much cheaper. This is because, most of the
    # time, there are no cells to be removed and no edges to be flipped. (The flip is
    # still a fairly expensive operation.)
    while True:
        idx = mesh.is_boundary_point
        points_new = mesh.points.copy()
        points_new[idx] = geo.boundary_step(points_new[idx].T).T
        mesh.points = points_new
        #
        num_removed_cells = mesh.remove_boundary_cells(
            lambda is_bdry_cell: mesh.compute_signed_cell_volumes(is_bdry_cell)
            < 1.0e-10
        )
        #
        # The flip has to come right after the boundary cell removal to prevent
        # "degenerate cell" errors.
        mesh.flip_until_delaunay(tol=flip_tol)
        #
        if num_removed_cells == 0:
            break

    # удалить все граничные ячейки, барицентры которых не находятся в геометрии
    mesh.remove_boundary_cells(
        lambda is_bdry_cell: geo.dist(mesh.compute_cell_centroids(is_bdry_cell).T) > 0.0
    )


def _recell(mesh, geo, flip_tol):
    mesh.remove_boundary_cells(
        lambda is_boundary_cell: geo.dist(mesh.compute_centroids(is_boundary_cell).T)
        > 0.0
    )
    mesh.remove_boundary_cells(
        lambda is_boundary_cell: mesh.compute_signed_cell_areas(is_boundary_cell)
        < 1.0e-10
    )
    mesh.flip_until_delaunay(tol=flip_tol)


def create_staggered_grid(h, bounding_box):
    x_step = h
    y_step = h * np.sqrt(3) / 2
    bb_width = bounding_box[1] - bounding_box[0]
    bb_height = bounding_box[3] - bounding_box[2]
    midpoint = [
        (bounding_box[0] + bounding_box[1]) / 2,
        (bounding_box[2] + bounding_box[3]) / 2,
    ]

    num_x_steps = int(bb_width / x_step)
    if num_x_steps % 2 == 1:
        num_x_steps -= 1
    num_y_steps = int(bb_height / y_step)
    if num_y_steps % 2 == 1:
        num_y_steps -= 1

    x2 = num_x_steps // 2
    y2 = num_y_steps // 2
    x, y = np.meshgrid(
        midpoint[0] + x_step * np.arange(-x2, x2 + 1),
        midpoint[1] + y_step * np.arange(-y2, y2 + 1),
    )
    offset = (y2 + 1) % 2
    x[offset::2] += h / 2

    out = np.column_stack([x.reshape(-1), y.reshape(-1)])

    n = 2 * (-(-y2 // 2))
    extra = np.empty((n, 2))
    extra[:, 0] = midpoint[0] - x_step * x2 - h / 2
    extra[:, 1] = midpoint[1] + y_step * np.arange(-y2 + offset, y2 + 1, 2)

    out = np.concatenate([out, extra])
    return out


def generate(
    geo,
    target_edge_size: Union[float, Callable],
    min_edge_size: int = 1,
    max_edge_size: int = 10,
    dist_fun_koef: float = 0.3,
    geo_min = 0,
    smoothing_method="distmesh",
    tol: float = 1.0e-5,
    random_seed: int = 0,
    need_show: bool = False,
    max_steps: int = 600,
    verbose: bool = True,
    flip_tol: float = 0.0,
    geo_cont = None,
):
    target_edge_size_function = (
        target_edge_size
        if callable(target_edge_size)
        else lambda pts: np.full(pts.shape[1], target_edge_size)
    )
    
    # Find h0 from edge_size (function)
    if callable(target_edge_size):
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = create_staggered_grid(h00, geo.bounding_box)
        # if 'list' in str(type(geo_min)):
        #     sizes = []
            # for mgeo in geo_min:
        if 'path' in str(type(geo_min)) or 'union' in str(type(geo_min)):
            target_edge_size_function = lambda pts: min_edge_size + dist_fun_koef * geo_min.dist(pts)
        else:
            koef = np.abs(geo_min.dist(min(pts, key = lambda x: x[0])))/max_edge_size
            target_edge_size_function = lambda pts: np.abs(geo_min.dist(pts)) / koef + min_edge_size
            
                # sizes.extend(edge_size_function(pts.T))
        # else:
        sizes = target_edge_size_function(pts.T)
        assert np.all(sizes > 0.0), "edge_size_function must be strictly positive."
        h0 = np.min(sizes)
    else:
        h0 = target_edge_size


    if random_seed is not None:
        np.random.seed(random_seed)

    pts = create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # удалить точки за пределами региона
    pts = pts[geo.dist(pts.T) < eps]

    # оценить функцию размера элемента, удалить точки в соответствии с ней
    alpha = 1.0 / target_edge_size_function(pts.T) ** 2
    rng = np.random.default_rng(random_seed)
    pts = pts[rng.random(pts.shape[0]) < alpha / np.max(alpha)]

    num_feature_points = len(geo.feature_points)
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = np.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = np.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = np.any(dist < ftol ** 2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = np.concatenate([geo.feature_points, pts])

    cells = _create_cells(pts, geo)
    mesh = meshplex.MeshTri(pts, cells)
    # When creating a mesh for the staggered grid, degenerate cells can very well occur
    # at the boundary, where points sit in a straight line. Remove those cells.
    mesh.remove_cells(mesh.q_radius_ratio < 1.0e-5)
    dim = 2
    mesh = distmesh_smoothing(
        mesh,
        geo,
        num_feature_points,
        target_edge_size_function,
        max_steps,
        tol,
        verbose,
        need_show,
        delta_t=0.2,
        f_scale=1 + 0.4 / 2 ** (dim - 1),  # from the original article
        flip_tol=flip_tol,
        
    )
    
    _points = mesh.points
    if geo_cont != None:
        contx = _points[geo_cont.dist(_points.T) < 1e-8]
        if len(geo_cont.feature_points) < len(contx):
            for _x in contx.tolist():
                if _x not in geo_cont.feature_points.tolist():
                    print(_x)
                    _points = np.delete(_points, np.where((_points == _x).all(axis=1))[0][0], axis=0)
                    
            cells = _create_cells(_points, geo)
            mesh = meshplex.MeshTri(_points, cells)
            mesh = distmesh_smoothing(
                    mesh,
                    geo,
                    num_feature_points,
                    target_edge_size_function,
                    5,
                    tol,
                    verbose,
                    need_show,
                    delta_t=0.2,
                    f_scale=1 + 0.4 / 2 ** (dim - 1),  # from the original article
                    flip_tol=flip_tol,
                    
                )
    points = mesh.points
    cells = mesh.cells("points")
    areas = mesh.compute_signed_cell_areas()
    #points, cells = optimesh.optimize_points_cells(points, cells, "cpt-linear-solve", 1.0e-5, 100)

    return points, cells, areas


def distmesh_smoothing(
    mesh,
    geo,
    num_feature_points,
    edge_size_function,
    max_steps,
    tol,
    verbose,
    need_show,
    delta_t,
    f_scale,    
    flip_tol=0.0,
    
):
    mesh.create_edges()
    
    angle_min = 0.523599
    angle_max = 1.658
    k = 0
    move2 = [0.0]
    with open('parameter.txt', 'w') as file:
        file.write(f"average_edges_size;bad_ratio;bad_angle\n")
    
    logger.info('Mesh quality parameters:')
    logger.info('min ce_ratio (covolume/edge_length ratios) = 1.0e-5')
    logger.info(f'min angles trias = {int(angle_min*180/3.14)}')
    logger.info(f'max angles trias = {int(angle_max*180/3.14)}')
    logger.info('target cell quality = 1')
    logger.info('\n')
    
    # plt.figure()
    # plt.ion()
    # fig, (ax1, ax2) = plt.subplots(1,2)

    
    
    while True:
        # print()
        # print(f"step {k}")
        if verbose:
            logger.info(f"step {k}")

        if k > max_steps:
            if verbose:
                logger.info(f"Exceeded max_steps ({max_steps}).")
            break

        k += 1

        if need_show:
            #print(f"max move: {math.sqrt(max(move2)):.3e}")
            show(mesh.points, mesh.cells["points"], geo)
            plt.savefig(f'P:/GTD/grid_maker/pict/mesh_{k}.png')
            plt.close()

        edges = mesh.edges["points"]

        edges_vec = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
        edge_lengths = np.sqrt(np.einsum("ij,ij->i", edges_vec, edges_vec))
        
        edges_vec /= edge_lengths[..., None]

        # оценка размеров элементов в средних точках линий
        edge_midpoints = (mesh.points[edges[:, 1]] + mesh.points[edges[:, 0]]) / 2
        p = edge_size_function(edge_midpoints.T)
        desired_lengths = (
            f_scale
            * p
            * np.sqrt(np.dot(edge_lengths, edge_lengths) / np.dot(p, p))
        )
        
        #koef = (desired_lengths + edge_lengths)/(2*desired_lengths)
        koef = 1
        force_abs = koef*(desired_lengths - edge_lengths)
        # учитывать только силы отталкивания
        force_abs[force_abs < 0.0] = 0.0

        # векторы силы
        force = edges_vec * force_abs[..., None]

        n = mesh.points.shape[0]
        # сила на точку
        force_per_point = np.array(
            [
                np.bincount(edges[:, 0], weights=-force[:, k], minlength=n)
                + np.bincount(edges[:, 1], weights=+force[:, k], minlength=n)
                for k in range(force.shape[1])
            ]
        ).T

        update = delta_t * force_per_point
        points_old = mesh.points.copy()
        points_new = mesh.points + update
        points_new[:num_feature_points] = mesh.points[:num_feature_points]
        
        
        mesh.points = points_new
        
        _recell_and_boundary_step(mesh, geo, flip_tol)
        diff = points_new - points_old
        
        move2 = np.einsum("ij,ij->i", diff, diff)
        
        
        # qual = [quality(mesh.points[i]) for i in mesh.cells("points")]
        # ax1.clear()
        # ax1.hist(qual, density=True, bins=30)
        # #ax1.axis("square")
        
        # ax2.clear();
        # ax2.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells("points"))
        # ax2.axis("square")
        
        
        # plt.draw()
        # plt.pause(0.1)

        no_qu_r_ratio = len(mesh.q_radius_ratio[abs(mesh.q_radius_ratio - 1) > 0.15])
        min_a = mesh.angles[0][mesh.angles[0] > angle_min]
        no_min_ang = len(mesh.angles[0]) -  len(min_a[min_a < angle_max])
        avrg_el = sum(edge_lengths)/len(edge_lengths)
        with open('parameter.txt', 'a') as file:
            file.write(f"{avrg_el};{no_qu_r_ratio};{no_min_ang}\n")
        if verbose:
            logger.info("max_move: {:.6e}".format(np.sqrt(np.max(move2))))
            logger.info('-'*64)
            logger.info(f"|{'Average edges size'.center(20)}|{'bad ratio'.center(20)}|{'bad angle'.center(20)}|")
            logger.info('-'*64)
            logger.info(f"|{str('%.3f' %avrg_el).center(20)}|{str(no_qu_r_ratio).center(20)}|{str(no_min_ang).center(20)}|")
            logger.info('-'*64)
            # logger.info(f'Average edges size: {"%.3f" %sum(edge_lengths)/len(edge_lengths)}')            
            # logger.info("bad ratio: {}".format(no_qu_r_ratio))
            # logger.info("bad angle: {}".format(no_min_ang))
        if np.all(move2 < tol ):
            break
    print(np.max(move2))
    print(k)
    # plt.ioff()
    # plt.show()
    
    logger.info(f"num steps:  {k}")
    mesh.remove_dangling_points()
    return mesh
