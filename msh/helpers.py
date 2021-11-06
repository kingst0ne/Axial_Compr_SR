
def show(pts, cells, geo, title=None, full_screen=True):
    import matplotlib.pyplot as plt

    eps = 1.0e-10
    is_inside = geo.dist(pts.T) < eps
    #plt.plot(pts[is_inside, 0], pts[is_inside, 1], ".")
    #plt.plot(pts[~is_inside, 0], pts[~is_inside, 1], ".", color="r")
    plt.triplot(pts[:, 0], pts[:, 1], cells)
    plt.axis("square")

    # show cells indices
    # for idx, barycenter in enumerate(numpy.sum(pts[cells], axis=1) / 3):
    #     plt.plot(*barycenter, "xk")
    #     plt.text(
    #         *barycenter, idx, horizontalalignment="center", verticalalignment="center"
    #     )

    # show node indices
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

