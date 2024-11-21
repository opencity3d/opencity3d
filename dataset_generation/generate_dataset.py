import pyrender
import trimesh
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm

import platform
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Create a scene
# file = "data/brooklyn-bridge.ply"
#file = "C:/MSC-Data/OpenCityData/brooklyn-bridge-obj/brooklyn-bridge.obj"
#file = "/home/bieriv/LangSplat/LangSplat/data/buenos-aires-2-smaller-mesh/buenos-aires-2.obj"
# file = "/home/bieriv/LangSplat/LangSplat/data/rotterdam/rotterdam.obj"
# file = "/home/bieriv/LangSplat/LangSplat/data/buenos-aires-squared/buenos-aires-squared-shifted.obj"
file = "/home/bieriv/LangSplat/LangSplat/data/ams/ams.glb"
# file = "/home/bieriv/LangSplat/LangSplat/data/delft/delft.glb"
# file = "/home/bieriv/LangSplat/LangSplat/data/ams/ams.glb"
#ile = "C:/MSC-Data/OpenCityData/brooklyn-bridge-ply/brooklyn-bridge.ply"
if False:
    import open3d as o3d
    mesh = o3d.io.read_triangle_model(file, True)
    # model = o3d.visualization.rendering.TriangleMeshModel(mesh)
    o3d.visualization.draw_geometries([mesh])
# output_path = "data/brooklyn-bridge"
# output_path = "data/buenos-aires-2-smaller-mesh-out
# put"
# output_path = "data/buenos-aires-2-smaller-mesh-output-low-cam"
# output_path = "data/buenos-aires-squared-output-v3"
output_path = "data/ams-output-v2"
continue_from_last = True
width = 384
height = 384

mesh_glb = trimesh.load(file)
# mesh_glb = trimesh.Scene(mesh_glb)
print(mesh_glb)

np.random.seed(42)

# define camera intrinsics
fx = 1200 // 2
fy = 1200  // 2
cx = 600 // 2
cy = 600 // 2
z_far = 2000
z_near = 10
# measure scene
intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                            [0, 0, 1]]) # define K matrix

x_grid_size = 500
z_grid_size = 500

# Create a renderer
scene = pyrender.Scene.from_trimesh_scene(mesh_glb, bg_color=[1.0, 1.0, 1.0])
ambient_intensity = 0.8  # Adjust intensity as needed
scene.ambient_light = np.array([ambient_intensity, ambient_intensity, ambient_intensity])

renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)


bounds  = scene.bounds
border = 10 #300
max_position_noise = 100
xrange = [45, 65] # vertical angle (90 means bird view / satellite and 0 means horizontal / street view)
yrange = [0, 360] # horizontal angle, we want to go round round round
height_range = [25, 100] #[500, 550] # [50, 100]
n_retries = 7 # 7
max_n_samples = 7500
orth_prob = 0.3


if not os.path.isdir(output_path):
    os.mkdir(output_path)
    os.mkdir(f"{output_path}/depth")
    os.mkdir(f"{output_path}/color")
    os.mkdir(f"{output_path}/pose")
    os.mkdir(f"{output_path}/intrinsic")

if continue_from_last: 
    existing = [int(f.split(".")[0]) for f in os.listdir(f"{output_path}/depth")]
    counter = max(existing) if len(existing) > 0 else 0
    print(f"Continuing from image {counter}")
else:
    counter = 0
reuse_pose = False
if not reuse_pose:
    for x_pos in tqdm(np.linspace(bounds[0,0]+border, bounds[1,0]-border, int(np.sqrt(max_n_samples)))):
        for z_pos in np.linspace(bounds[0,2]+border, bounds[1,2]-border, int(np.sqrt(max_n_samples))):
            if os.path.exists(f"{output_path}/depth/{counter}.npy"):
                counter += 1
                continue
            for i in range(n_retries):
                random_height = np.random.randint(height_range[0], height_range[1])
                y_angle = np.random.randint(yrange[0], yrange[1])
                x_angle = 90 if np.random.rand() < orth_prob else np.random.randint(xrange[0], xrange[1])
                z_angle = 0
                x_pos += np.random.randint(-max_position_noise//2, max_position_noise//2)
                z_pos += np.random.randint(-max_position_noise//2, max_position_noise//2)
                camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=z_far, znear=z_near)

                min_distance = -1 
                while min_distance < 50: 
                    if min_distance > 0:
                        print(f"min_distance: {min_distance}")
                    # move the camera up until there is nothing too close to it
                    rotation_matrix = R.from_euler(seq="yxz", angles=[y_angle,x_angle,z_angle], degrees=True).as_matrix()
                    extrinsic_matrix = np.block([[rotation_matrix, np.array([0, 0, 0]).reshape(-1, 1)],
                                                [0, 0, 0, 1]])
                    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
                    # set camera pos instead of world pos for simplicity
                    extrinsic_matrix[:3, 3] = [x_pos, random_height, z_pos]
                    
                    assert np.isclose(np.linalg.det(extrinsic_matrix), 1), f"Determinant of extrinsic matrix is not 1 but {np.linalg.det(extrinsic_matrix)}"

                    # Render the scene
                    added_node = scene.add(camera, pose=extrinsic_matrix)
                    scene.main_camera_node = added_node

                    color, depth = renderer.render(scene) 
                    if (depth > z_near).sum() == 0:
                        break
                    min_distance = np.min(depth[depth > z_near]) 
                    random_height += 25
                
                if np.sum(depth < z_near) < 0.20 * depth.size:
                    np.save(f"{output_path}/depth/{counter}.npy", depth.astype(np.float16))
                    imageio.imwrite(f"{output_path}/color/{counter}.jpg", color)
                    np.savetxt(f"{output_path}/pose/{counter}.txt", extrinsic_matrix, fmt='%f')
                    counter += 1
                    break
                elif i == n_retries - 1:
                    print("Failed to render image after max retries")
else:
    pose_paths = [f"{output_path}/pose/{f}" for f in os.listdir(f"{output_path}/pose")]
    poses = list(map(np.loadtxt, pose_paths))
    for pose, path in tqdm(zip(poses, pose_paths), total=len(poses)):
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=z_far, znear=z_near)
        added_node = scene.add(camera, pose=pose)
        scene.main_camera_node = added_node
        color, depth = renderer.render(scene)  
        np.save(path.replace("/pose/", "/depth/").replace(".txt", ".npy"), depth)
        imageio.imwrite(path.replace("/pose/", "/color/").replace(".txt",".jpg"), color)
np.savetxt(f"{output_path}/intrinsic/intrinsic_color.txt", intrinsic_matrix, fmt='%f')
np.savetxt(f"{output_path}/intrinsic/projection_matrix.txt", camera.get_projection_matrix(width, height), fmt='%f')