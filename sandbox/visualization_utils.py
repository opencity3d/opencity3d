import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import matplotlib

def set_matplotlib_defaults():
    font = {'family' : 'Raleway', 'weight' : 'normal', 'size' : '10'}
    fp = matplotlib.font_manager.FontProperties(**font)
    matplotlib.rc('font', **font)
    matplotlib.rcParams['xtick.labelsize'] = 9 # 6.5
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.width'] = 0.25
    matplotlib.rcParams['ytick.labelsize'] = 9
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['ytick.major.width'] = 0.5
    matplotlib.rcParams['ytick.minor.width'] = 0.25
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['axes.labelsize'] = 11
    matplotlib.rcParams['axes.linewidth'] = 0.5
    matplotlib.rcParams['lines.linewidth'] = 0.5
    matplotlib.rcParams['lines.markersize'] = 2
    matplotlib.rcParams['lines.markeredgewidth'] = 0.5
    matplotlib.rcParams['figure.max_open_warning'] = False



def grayscale_to_plasma(image, normalize=True):
    image = np.array(image)
    colormap = plt.get_cmap('magma')
    if normalize:
        image = image / image.max()
    return colormap(image)[...,:3]


def visualize(pcd, sim, query, normalize=True, histogram=False):
    pcd = deepcopy(pcd)
    pcd_colors = deepcopy(pcd.colors)

    def switch_coloring(vis):
        print("Switching coloring")
        pcd.colors = pcd_colors if pcd.colors != pcd_colors else semantic_colors
        vis.update_geometry(pcd)
        return False
    
    def set_pos(vis):
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("C:/Users/Valentin\PycharmProjects\OpenCity\sandbox\ScreenCamera_2024-05-26-16-43-45.json".replace('\\',"/"))
        ctr.convert_from_pinhole_camera_parameters(parameters, True)
        vis.update_renderer()
        vis.update_geometry(pcd)
    
    colors = grayscale_to_plasma(sim, normalize=normalize)
    colors[np.asarray(pcd_colors).sum(axis=1) == 0] = 0
    if histogram:
        plt.hist(sim, bins=100)
        plt.title(query)
        print(colors.max(), colors.min())
        plt.show()
    semantic_colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd.colors = semantic_colors

    assert pcd.has_colors()

    o3d.visualization.draw_geometries_with_key_callbacks([pcd],
                                                              {ord("K"):switch_coloring, ord("Q"): set_pos},
                                                              window_name = query)