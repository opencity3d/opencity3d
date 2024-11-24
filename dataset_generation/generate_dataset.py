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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Render a set of 2D images from a 3D mesh')
    parser.add_argument('--input-mesh', type=str, required=True, help='Path to the mesh file (glb or any other compatible with trimesh)')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--width', type=int, default=384, help='Width of the image')
    parser.add_argument('--height', type=int, default=384, help='Height of the image')
    parser.add_argument('--approx-n-samples', type=int, default=10000, help='Approximate number of samples to take')
    parser.add_argument('--skip-existing', action='store_true', help='Skip rendering images that already exist in the output directory. Leads to inconsistencies if the number of samples is not the same as previously')
    return parser.parse_args()


def render_to_directory(file, output_path, width=384, height=384, max_n_samples=10000, skip_existing=False):
    """Renders a 3D mesh to 2D images, saving the output to the specified directory alongside depth maps, poses, intrinsics, ...
    Samples camera poses from a grid with random offset. Height is chosen randomly within range. 
    However, the camera is moved upwards if there is any object too close to it.
    Parameters:
        file (str): Path to the input 3D mesh file.
        output_path (str): Directory where the rendered outputs will be saved.
        width (int, optional): Width of the rendered images. Default is 384.
        height (int, optional): Height of the rendered images. Default is 384.
        max_n_samples (int, optional): Maximum number of samples to generate. Default is 10000.
        skip_existing (bool, optional): If True, skips rendering if the output files already exist. Default is False.
    Outputs:
        Saves the following files in the specified output directory:
        - Color images in the 'color' subdirectory.
        - Depth maps in the 'depth' subdirectory.
        - Camera poses in the 'pose' subdirectory.
        - Intrinsic matrices in the 'intrinsic' subdirectory.
    """
    mesh_glb = trimesh.load(file)
    print(f"Read mesh: ", mesh_glb)

    np.random.seed(42)

    # define camera intrinsics
    fx = 1200 // 2
    fy = 1200  // 2
    cx = 600 // 2
    cy = 600 // 2
    z_far = 2000
    z_near = 10
    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    # Create a renderer
    scene = pyrender.Scene.from_trimesh_scene(mesh_glb, bg_color=[1.0, 1.0, 1.0])
    ambient_intensity = 0.8 
    scene.ambient_light = np.array([ambient_intensity, ambient_intensity, ambient_intensity])
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)


    bounds  = scene.bounds
    border = 10 # how much to stay away from the border with the camera position (on horizontal plane)
    max_position_noise = 100 # how much we can move the camera around randomly (on horizontal plane)
    xrange = [45, 65] # vertical angle (90 means bird view / satellite and 0 means horizontal / street view)
    yrange = [0, 360] # horizontal angle, we want to go round round round
    height_range = [scene.bounds[0][1] + 15, scene.bounds[0][1] + 100] # choose the camera height in given range over ground
    n_retries = 7 # how often to move camera upwards before giving up
    orth_prob = 0.3 # probability of choosing an orthogonal birds-eye view
    min_distance_to_structure = 50 # how close (m) we can be to the closest structure (depth image)


    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/depth", exist_ok=True)
    os.makedirs(f"{output_path}/color", exist_ok=True)
    os.makedirs(f"{output_path}/pose", exist_ok=True)
    os.makedirs(f"{output_path}/intrinsic", exist_ok=True)

    counter = 0
    for x_pos in tqdm(np.linspace(bounds[0,0]+border, bounds[1,0]-border, int(np.sqrt(max_n_samples)))):
        for z_pos in np.linspace(bounds[0,2]+border, bounds[1,2]-border, int(np.sqrt(max_n_samples))):
            if skip_existing and os.path.exists(f"{output_path}/depth/{counter}.npy"):
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
                while min_distance < min_distance_to_structure: 
                    # move the camera up until there is nothing too close to it
                    rotation_matrix = R.from_euler(seq="yxz", angles=[y_angle,x_angle,z_angle], degrees=True).as_matrix()
                    extrinsic_matrix = np.block([[rotation_matrix, np.array([0, 0, 0]).reshape(-1, 1)],
                                                [0, 0, 0, 1]])
                    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
                    # set camera pos instead of world pos for simplicity
                    extrinsic_matrix[:3, 3] = [x_pos, random_height, z_pos]
                    
                    assert np.isclose(np.linalg.det(extrinsic_matrix), 1), f"Determinant of extrinsic matrix is not 1 but {np.linalg.det(extrinsic_matrix)}"

                    # Render an image
                    added_node = scene.add(camera, pose=extrinsic_matrix)
                    scene.main_camera_node = added_node

                    color, depth = renderer.render(scene) 
                    if (depth > z_near).sum() == 0:
                        break
                    min_distance = np.min(depth[depth > z_near]) 
                    random_height += 25
                
                if np.sum(depth < z_near) < 0.20 * depth.size:
                    # only save if at least 80% of the image is valid
                    np.save(f"{output_path}/depth/{counter}.npy", depth.astype(np.float16))
                    imageio.imwrite(f"{output_path}/color/{counter}.jpg", color)
                    np.savetxt(f"{output_path}/pose/{counter}.txt", extrinsic_matrix, fmt='%f')
                    counter += 1
                    break
                elif i == n_retries - 1:
                    pass
                    # print("Failed to render image after max retries")
    
    np.savetxt(f"{output_path}/intrinsic/intrinsic_color.txt", intrinsic_matrix, fmt='%f')
    np.savetxt(f"{output_path}/intrinsic/projection_matrix.txt", camera.get_projection_matrix(width, height), fmt='%f')


if __name__ == "__main__":
    args = parse_args()
    render_to_directory(args.input_mesh, args.output_dir, args.width, args.height, args.approx_n_samples, args.skip_existing)
    