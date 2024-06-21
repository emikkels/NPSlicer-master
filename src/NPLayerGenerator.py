OUTPUT_DIR = 'output'
CACHE_DIR = 'cache'


import os
import math
import shutil
import time
import multiprocessing as mp

import numpy as np

import open3d as o3d
import trimesh
from trimesh.voxel.creation import local_voxelize


import distinctipy  # For distinct random color generation

import matplotlib.pyplot as plt
import matplotlib.cm as cm        
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle


from numba import jit

from scipy import ndimage
from scipy.spatial import  KDTree


class NPLayerGenerator():
    def __init__(self, input_path:str, data_path:str=None, pitch:float=1.0, layer_thickness_mm:float=1.0) -> None:
        self.filename = os.path.basename(input_path)[:-4]
        self.input_path = input_path
        self.pitch = pitch
        self.layer_thickness_mm = layer_thickness_mm
        
        if data_path is not None:
            self.cache_dir = f'{data_path}\\{self.filename}\\cache\\p{self.pitch}'
            self.output_dir = f'{data_path}\\{self.filename}\\output\\p{self.pitch}\\lt{layer_thickness_mm}mm'    
        else:    
            self.cache_dir = f'{CACHE_DIR}\\{self.filename}\\p{self.pitch}'
            self.output_dir = f'{OUTPUT_DIR}\\{self.filename}\\p{self.pitch}'
        self.dense_path = f'{self.cache_dir}\\dense_{self.pitch}.npy'
            
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir) 
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
       
        self.mesh = trimesh.load_mesh(input_path)
        
        # Ensure mesh bounding box is centered at origin
        self.mesh.apply_translation(-self.mesh.bounding_box.centroid)
        # Move to xy-plane
        self.mesh.apply_translation([0, 0, -self.mesh.bounds[0][2]])

        self.load_or_generate_dense()     
        
        self.shape = self.dense.shape

    def load_or_generate_dense(self) -> None:

        try:
            self.dense = np.load(self.dense_path)
            print('Loaded dense from cache')
        except:
            self.voxelize()
            self.dense = np.load(self.dense_path)
    
        self.filled = np.count_nonzero(self.dense)
        print(f'Voxel grid shape: {self.dense.shape}, {np.count_nonzero(self.dense):,} filled voxels.')

    def voxelize(self, method:str='trimesh_surface_fill') -> None:
        print(f'Voxelizing mesh with {self.pitch}mm pitch')
        
        mesh_copy = self.mesh.copy()
        if method == 'trimesh':
            radius = int(np.sqrt(2) * np.max(self.mesh.bounds[1] - self.mesh.bounds[0])/self.pitch * .5) + 5    
            voxels = local_voxelize(mesh_copy, self.mesh.bounding_box.centroid, self.pitch, radius, fill=True)
            dense = voxels.matrix
            
        if method == 'trimesh_surface_fill':
            # Use surface voxelization and fill holes using binary_fill_holes: # https://github.com/mikedh/trimesh/issues/200
            voxelized_surface = mesh_copy.voxelized(pitch=self.pitch)
            dense_surface = voxelized_surface.matrix
            dense = ndimage.binary_fill_holes(dense_surface)
            
        # Remove padding of False elements around the voxel grid
        true_indices = np.argwhere(dense)
        min_x, min_y, min_z = np.min(true_indices, axis=0)
        max_x, max_y, max_z = np.max(true_indices, axis=0)
        dense = dense[min_x:max_x+1, min_y:max_y+1, min_z:max_z]
        

        print(f'Voxelization of {self.filename} complete.')

        # Save to cache for faster loading next time
        np.save(self.dense_path, dense)
        print('Saved dense to cache')
        
    def generate_overhang_mask(self, dense:np.ndarray,  t:float=3.) -> np.ndarray:
        self.max_t_mm = t
        self.t_index_offset = math.ceil(t / self.pitch) # Number of voxels to add to the overhang mask in z+ direction
        print('Finding overhanging voxels')
        overhang_mask = find_overhang_voxels(dense)
        overhang_mask[:, :, 0] = False

        if t > 0:
            t_voxels = int(t / self.pitch)
            z_max = overhang_mask.shape[2]-1
            for z in range(z_max, 1, -1):
                overhang_2d = overhang_mask[:, :, z]
                if np.any(overhang_2d):
                    overhang_mask[:, :, z:min(z_max, z+t_voxels)] = overhang_mask[:, :, z:min(z_max, z+t_voxels)] + overhang_2d[:, :, np.newaxis]
        
        # Ensure top and bootom of each overhang cluster is flat (to avoid collapsing layers)
        clusters, n_clusters = ndimage.label(overhang_mask)
        for i in range(n_clusters):
            cluster = (clusters == i+1) # 3D mask of the current cluster
            # Find the top and bottom z-index of the cluster
            top = np.max(np.argwhere(cluster), axis=0)[2]
            bottom = np.min(np.argwhere(cluster), axis=0)[2]
            
            # Fill in the cluster between top and bottom
            unique_xy = np.unique(np.argwhere(cluster)[:, 0:2], axis=0)
            for x, y in unique_xy:
                overhang_mask[x, y, bottom:top+1] = True
        return overhang_mask & dense  

    def generate_scalar_field(self, alpha:float=45., t:float=1., pad:int=0, o_pad:int=0) -> np.ndarray:
        
        self.alpha = alpha
        self.shape = self.dense.shape
        self.overhang_mask = self.generate_overhang_mask(self.dense, t=t)
        
        
        self.unpadded_dense = np.copy(self.dense)
        if pad > 0:
            for i in range(pad):
                padded_surface = pad_surface(self.dense)
                padded_dense = padded_surface
                padded_dense[1:-1, 1:-1, 1:-1] += self.dense
                self.dense = padded_dense
                self.overhang_mask = np.pad(self.overhang_mask, 1, mode='constant', constant_values=False)
            self.shape = self.dense.shape

        # Pad overhang mask
        if o_pad > 0:
            o_opad_layers = np.max([o_pad, pad])
            method = 'not_top'
        else:
            o_opad_layers = pad
            method = 'xy'
            
        for i in range(o_opad_layers):
            padded_surface = pad_surface(self.overhang_mask, method=method) 
            padded_overhang_mask = padded_surface
            padded_overhang_mask[1:-1, 1:-1, 1:-1] += self.overhang_mask
            
            self.overhang_mask = padded_overhang_mask[1:-1, 1:-1, 1:-1]
            self.overhang_mask = self.overhang_mask & self.dense      # Filter overhang mask to only keep filled voxels
            
        self.transform = np.array([self.shape[0]*self.pitch/2 - self.pitch/2, 
                                   self.shape[1]*self.pitch/2 - self.pitch/2, 
                                   pad*self.pitch + 0.01])

        scalar_field = np.zeros_like(self.dense, dtype=np.float64)
        
        extended_overhang_mask = self.overhang_mask | ~self.dense # Extend overhang mask to include all empty voxels  
        
        supported_mask = (self.dense & ~extended_overhang_mask)
        
        # Set scalar field values based on z-height to create initial planar layer structure
        n_z = scalar_field.shape[2]
        self.delta_iso = self.pitch/self.layer_thickness_mm  # Iso value difference between each voxel. Spacing is 1 for each layer
        for i in range(n_z):    
            scalar_field[:, :, i] = i * self.delta_iso * 1  # Set scalar field values based on z-height to create initial planar layer structure
        self.planar_scalar_field = scalar_field.copy()
        
        # k is the multiplier for the distance to nearest overhanging voxel. Could use other equations here to create curved layers as a function of overhang distance, position etc.
        # but then k is no longer constant and must be calculated inside the loop
        
        k = np.tan(np.radians(alpha)) * self.delta_iso  # Calculate the slope for the overhang offsets
        overhang_distance_to_supported = None
        is_overhang = False
        for i in range(n_z):
            if  i > 0 and not np.any(supported_mask[:, :, i]): # If no voxels are supported, use the previous layer's overhang distance field
                if overhang_distance_to_supported is None:
                    distance_to_supported = ndimage.distance_transform_edt(~supported_mask[:, :, i-1])
                    overhang_distance_to_supported = distance_to_supported * extended_overhang_mask[:, :, i].astype(int)  
                # else: Do nothing, distance fields remain the same as the last layer where voxels were supported
            elif np.any(self.overhang_mask[:, :, i]):
                is_overhang = True
                distance_to_supported = ndimage.distance_transform_edt(~supported_mask[:, :, i])  # Compute distance distance field of distance to supported voxels 
                overhang_distance_to_supported = distance_to_supported * extended_overhang_mask[:, :, i].astype(int)  # Filter distance field to only keep distances for overhanging voxels
            else: # Manually set distance field to 0 if no overhanging voxels in the current layer
                distance_to_supported = np.zeros_like(self.dense[:, :, i], dtype=np.float64)
                overhang_distance_to_supported = np.zeros_like(self.dense[:, :, i], dtype=np.float64)

            if np.any(self.overhang_mask[:, :, i]):  # If there are overhanging voxels in the current layer
                scalar_field[:, :, i] = scalar_field[:, :, i] + k * overhang_distance_to_supported  # Add the distance field * k to the scalar field
            else:   
                if is_overhang: # End of overhang region, add an offset to all subsequent layers to ensure proper layer ordering
                    is_overhang = False
                    prev_layer = scalar_field[:, :, i-1] * self.dense[:, :, i-1]
                    prev_layer_max = np.max(prev_layer)
                    prev_layer_min = np.min(prev_layer[prev_layer > 0])
                    scalar_field_offset = np.ceil(prev_layer_max / self.delta_iso - 1) * self.delta_iso - prev_layer_min
                    
                    scalar_field[:, :, i:] += scalar_field_offset 
            
            # Visualize 2d distance field, 2d overhang mask and 2d supported mask in one figure
            if False and i%5 == 0:
                fig, ax = plt.subplots(1, 6)
                fig.set_size_inches(30, 10)
                
                fig.suptitle(f'Layer {i}')
                
                ax[0].imshow(self.dense[:, :, i], cmap='binary')
                ax[0].set_title('Dense')
                ax[1].imshow(self.overhang_mask[:, :, i], cmap='Reds')
                ax[1].set_title('Overhang')
                ax[2].imshow(supported_mask[:, :, i], cmap='Greens')
                ax[2].set_title('Supported')
                ax[3].imshow(distance_to_supported, cmap='jet')
                ax[3].set_title('Distance field')
                ax[4].imshow(overhang_distance_to_supported, cmap='jet')
                ax[4].set_title('Overhang distance field')
                ax[5].imshow(scalar_field[self.shape[0]//2, :, :]*self.dense[self.shape[0]//2, :, :], cmap='jet')
                ax[5].set_title('Scalar field')

                # fig.show()
                fig.savefig(f'{self.output_dir}\\distance_field_layer_{i}.png')
                plt.close(fig)
                         
        self.scalar_field = scalar_field
        return scalar_field
       
  
    def generate_isosurface(self, scalar_field:np.ndarray, isovalue:float, n:int=1, c:float=1.0, trim:bool=True, filter:bool=True) -> trimesh.Trimesh:
        if VERBOSE:
            print(f'\n      Creating isosurface at isovalue {isovalue}')
        
        # Create a flat grid mesh and set the z-values to intepolated values
        z_coordinates = interpolate_z_crossing(scalar_field, isovalue)
        

        # Skip layer if no positive z-values
        if np.all(z_coordinates < 0):
            # print('No positive z-values')
            return None
        

        # Check if all positive z-values are equal, which means the current layer is planar
        if np.all(z_coordinates[z_coordinates > 0] == z_coordinates[z_coordinates > 0][0]):
            pass
            # if VERBOSE:
            #     print('Planar layer, no smoothing required')
        else:
            if n > 0:   # Smooth nonplanar layers using n nearest neighbors
                z_coordinates = smooth_z_coordinates2(z_coordinates, n, c)
        
              
        vertices, triangles = get_grid_mesh(z_coordinates)
        

        vertices = flatten_negative_vertices(vertices)
        
        # Translate vertices to real world coordinates
        vertices = vertices * self.pitch - self.transform
        
        # Boolean intersection with stl mesh
        layer_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        if trim:
            max_area = self.pitch**2    # Maximum area triangle to keep in the intersected mesh
            layer_mesh = mesh_intersection(layer_mesh, self.mesh, self.alpha, max_area, filter=filter)
        if layer_mesh.is_empty:
            return None
        

        print(f'Finished creating isosurface at isovalue {isovalue} with {len(layer_mesh.vertices)} vertices and {len(layer_mesh.faces)} faces. z_min = {np.round(np.min(layer_mesh.vertices[:, 2]), 2)}, z_max = {np.round(np.max(layer_mesh.vertices[:, 2]), 2)}')
        
        return layer_mesh

    def get_isosurfaces_mp(self, scalar_field:np.ndarray, isovalues:int|float|list|tuple|np.ndarray|None=None, n:int=0, c:float=1, export:bool=True) -> list[trimesh.Trimesh]:
        
        if not isinstance(isovalues, np.ndarray):
            if isovalues is None:
                isovalues = np.arange(1, np.max(scalar_field)+1, 1)
            elif isinstance(isovalues, (int, float)):
                isovalues = np.array([isovalues])
            elif isinstance(isovalues, (list, tuple)):
                isovalues = np.array(isovalues)
            else:
                raise ValueError('Invalid type for isovalues, must be None (=all), int, float, list, tuple or ndarray')


        isovalues = isovalues[isovalues <= np.max(scalar_field)]
        layer_meshes = []


        if len(isovalues) == 1:
            layer_meshes.append(self.generate_isosurface(scalar_field, isovalues[0], n, c))
        
        else:  # Parallelize isosurface generation for multiple isovalues 
            n_processes = mp.cpu_count() - 2  
            with mp.Pool(n_processes) as pool:
                print(f'Generating {len(isovalues)} isosurfaces using {n_processes} processes')
                results = pool.starmap(self.generate_isosurface, [(scalar_field, isovalue, n, c) for isovalue in isovalues])
                for res in results:
                    if res is not None:
                        layer_meshes.append(res)

        if export:
            shutil.rmtree(self.output_dir)  # Clear output directory before exporting new surfaces
            os.makedirs(self.output_dir)

        for i, layer_mesh in enumerate(layer_meshes):
            if export:
                if VERBOSE:
                    print(f'Exporting layer_{i}.stl with {len(layer_mesh.vertices)} vertices and {len(layer_mesh.faces)} faces')
                layer_mesh.export(f'{self.output_dir}\\layer_{i}.stl', file_type='stl_ascii')
        
        print(f'Generated {len(layer_meshes)} isosurfaces.')
        if export:
            print(f'Surfaces saved to {self.output_dir}')
        return layer_meshes
       
    def trim(self, meshes:list[trimesh.Trimesh], export:bool=False, min_layer_thickness:float|None=None, tol:float=0.01) -> list[trimesh.Trimesh]:
        if min_layer_thickness is None:
            min_layer_thickness = self.layer_thickness_mm
        
        
        layers_to_trim = []
        for i, mesh in enumerate(meshes):
            if i == 0:
                prev_zmax = np.max(mesh.vertices[:, 2])
                continue
            zmax = np.max(mesh.vertices[:, 2])
            if zmax - prev_zmax < min_layer_thickness - tol:
                layers_to_trim.append(i-1)
            prev_zmax = zmax
        
        # Isolate clusters of adjacent layers
        layer_clusters = np.split(layers_to_trim, np.where(np.diff(layers_to_trim) != 1)[0]+1)
        
        trimmed_meshes = np.copy(meshes)
        if len(layer_clusters) == 0:
            return trimmed_meshes # No trimming required
        
        for cluster in layer_clusters:
            z_start = np.max(meshes[cluster[0]].vertices[:, 2])
            for layer in cluster:
                trimmed_meshes[layer] = trimesh.intersections.slice_mesh_plane(meshes[layer], [0, 0, -1], [0, 0, z_start])
                if trimmed_meshes[layer].is_empty:
                    print(f'Empty mesh for layer {layer} after slice with plane at z = {z_start}')
                    trimmed_meshes[layer] = None     
        
            # Insert copies of first layer after transition to replace the trimmed regions
            z_insert_heights = np.arange(z_start + self.layer_thickness_mm, 
                                         z_start + self.pitch, 
                                         self.layer_thickness_mm)
            
            for i, z in enumerate(z_insert_heights):
                vertices, triangles = get_grid_mesh(np.zeros_like(self.scalar_field[:, :, 0]))
                vertices = vertices * self.pitch - self.transform
                vertices[:, 2] = z
                replacement_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                replacement_mesh = mesh_intersection(replacement_mesh, self.mesh, self.alpha, self.pitch**2, filter=False, show=False)
                trimmed_meshes[cluster[-1]-len(z_insert_heights)+i] = replacement_mesh

        trimmed_meshes = [mesh for mesh in trimmed_meshes if mesh is not None and not mesh.is_empty]

        if export:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)
            for i, mesh in enumerate(trimmed_meshes):
                print(f'Exporting layer_{i}.stl with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces')
                mesh.export(f'{self.output_dir}\\layer_{i}.stl')
        
        return trimmed_meshes

    ### VISUALIZATION FUNCTIONS ###
    def show_voxelgrid(self, highlight_voxelgrid:np.ndarray, crop:bool=False) -> None:

        def tensor_to_point_cloud(tensor):  # Helper function
            points = []
            for x in range(tensor.shape[0]):
                for y in range(tensor.shape[1]):
                    for z in range(tensor.shape[2]):
                        if tensor[x, y, z]:  # Voxel is present
                            points.append([x, y, z])

            return np.array(points) * self.pitch # Scale point by pitch to get real world coordinates
    
        filled_voxelgrid = self.dense
        if crop:
            filled_voxelgrid = self.dense[0:filled_voxelgrid.shape[0], 0:filled_voxelgrid.shape[1]//2, 0:filled_voxelgrid.shape[2]]
            highlight_voxelgrid = highlight_voxelgrid[0:highlight_voxelgrid.shape[0], 0:highlight_voxelgrid.shape[1]//2, 0:highlight_voxelgrid.shape[2]]

        highlight_points = tensor_to_point_cloud(highlight_voxelgrid)
        filled_filtered = filled_voxelgrid * ~highlight_voxelgrid
        filled_points = tensor_to_point_cloud(filled_filtered)
        
        # points = self.tensor_to_point_cloud(self.dense)
        
        filled_pcd = o3d.geometry.PointCloud()
        filled_pcd.points = o3d.utility.Vector3dVector(filled_points)
        highlight_pcd = o3d.geometry.PointCloud()
        highlight_pcd.points = o3d.utility.Vector3dVector(highlight_points)
        
        filled_colors = np.zeros_like(filled_points)
        filled_colors[:, 0] = 0.5
        filled_colors[:, 1] = 0.5
        filled_colors[:, 2] = 0.5
        filled_pcd.colors = o3d.utility.Vector3dVector(filled_colors)

        highlight_colors = np.zeros_like(highlight_points)
        highlight_colors[:, 0] = 1
        highlight_colors[:, 1] = 0
        highlight_colors[:, 2] = 0
        highlight_pcd.colors = o3d.utility.Vector3dVector(highlight_colors)
        
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        vis.add_geometry(filled_pcd)
        vis.add_geometry(highlight_pcd)
        
        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, 1, 0])
        view_ctl.set_up([0, 0, 1])

        vis.poll_events()
        vis.update_renderer()

        vis.run()

    def show_scalar_field(self, scalar_field:np.ndarray, crop:bool=False, layer_thickness_mm:float|None=None, show_mesh:bool=False) -> None:
        # Visualize the scalar field

        # Create a point cloud for visualization
        vis_dense = np.copy(self.dense)  # copy of the dense voxel grid for visualization
        if crop:
            vis_dense[:,0:vis_dense.shape[0]//2, :] = False    #   YZ-plane cross section

        surface_voxels = find_surface_voxels(vis_dense)
        scalar_field = scalar_field * vis_dense
        scalar_field = scalar_field*surface_voxels
        
        # Calculate number of layers in the scalar field given the layer thickness 
        s_max = np.max(scalar_field)
        if layer_thickness_mm is not None:
            n = int(np.ceil(s_max))
        else:
            n = vis_dense.shape[2]
        colormap = generate_random_cmap(n)
            
        colors = colormap(scalar_field[vis_dense]/s_max)[:, :3]  # Map scalar field values to colors
        # colors = cm.jet(scalar_field[vis_dense]/np.max(scalar_field[vis_dense]))[:, :3]  # Map scalar field values to colors
        # colors = cm.tab20(scalar_field[vis_dense]/np.max(scalar_field[vis_dense]))[:, :3]  # Map scalar field values to colors
        # colors = cm.viridis(scalar_field[vis_dense]/np.max(scalar_field[vis_dense]))[:, :3]  # Map scalar field values to colors

        # Transform points from voxel grid to real world coordinates
        points = np.argwhere(vis_dense).astype(np.float64)
        points = points * self.pitch
        points = points - self.transform


        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()
       
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.pitch)
        vis.add_geometry(voxel_grid)

        if show_mesh:
            mesh = self.mesh.as_open3d
            mesh.compute_vertex_normals()
            vis.add_geometry(mesh)

        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, 1, 0])
        view_ctl.set_up([0, 0, 1])

        vis.run()

    def show_isosurfaces(self, meshes:list[trimesh.Trimesh]|None=None, crop:bool=False, color:str='uniform', show_input_mesh:bool=False, reduce:float=0.0) -> None:

        # Create a Visualizer object
        vis = o3d.visualization.Visualizer()
        scale = 0.85
        width = int(scale*2560)
        height = int(scale*1440)
        vis.create_window(window_name='Isosurfaces', width=width, height=height, visible=True)
        # vis.create_window()
        geometries = []

        if meshes is None:
            meshes = self.load_isosurfaces()
            for i, mesh in enumerate(meshes):
                if reduce > 0:
                    mesh = mesh.simplify_quadric_decimation(len(mesh.faces) * reduce)
                mesh_o3d = mesh.as_open3d
                mesh_o3d.compute_vertex_normals() 
                if color == 'uniform':
                    mesh_o3d.paint_uniform_color([0.5, 0.5, 0.5])
                elif color == 'random':
                    mesh_o3d.paint_uniform_color(np.random.rand(3))
                elif color == 'jet':
                    mesh_o3d.paint_uniform_color(cm.jet(i/len(meshes))[:3])
            
                
                geometries.append(mesh_o3d)
                
        else:

            for i, mesh in enumerate(meshes):
                if isinstance(mesh, trimesh.Trimesh):   # Convert trimesh to open3d mesh
                    mesh = mesh.as_open3d
                mesh.compute_vertex_normals()
                if color == 'uniform':
                    mesh.paint_uniform_color([0.8, 0.8, 0.8])
                elif color == 'random':
                    mesh.paint_uniform_color(np.random.rand(3))
                elif color == 'jet':
                    mesh.paint_uniform_color(cm.jet(i/len(meshes))[:3])

                geometries.append(mesh)
        
        if show_input_mesh:
            mesh = self.mesh.as_open3d
            mesh.compute_vertex_normals()
            geometries.append(mesh)
        
        if crop:
            min_bounds, max_bounds = self.mesh.bounds
            min_bounds_vis = np.array([min_bounds[0], 0, min_bounds[2]])
            max_bounds_vis = np.array(max_bounds)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds_vis, max_bounds_vis)
            new_geometries = []
            for g in geometries:
                new_geometries.append(o3d.geometry.TriangleMesh.crop(g, bbox))
            geometries = new_geometries
        
        # Set the render options
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True      
        opt.mesh_show_wireframe = True
        opt.background_color = np.array([0.3, 0.3, 0.3])    # Dark gray background
        opt.light_on = True
        
        # Set the view control
        vis.get_view_control().set_front([ 0.34184232676861093, -0.87537015679537977, 0.3418638796677046 ])  
        vis.get_view_control().set_up([ -0.095109739692807116, 0.32968469192562944, 0.93928810347276803 ])
        vis.poll_events()
        vis.update_renderer()

        for g in geometries:
            vis.add_geometry(g)
            vis.get_view_control().set_front([ 0.34184232676861093, -0.87537015679537977, 0.3418638796677046 ])  
            vis.get_view_control().set_up([ -0.095109739692807116, 0.32968469192562944, 0.93928810347276803 ])
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.005)
        

        vis.poll_events()
        vis.update_renderer()
        # Run the visualizer
        vis.run()
        vis.destroy_window()
    
    def load_isosurfaces(self, path:str|None=None) -> list[trimesh.Trimesh]:
        if path is None:
            path = self.output_dir
            files = os.listdir(path)
            files = [file for file in files if file.startswith('layer')]
        else:
            files = os.listdir(path)
        layer_numbers = [int(file.split('_')[1].split('.')[0]) for file in files]
        files = [file for _, file in sorted(zip(layer_numbers, files))]
        meshes = []
        for file in files:
            if file.endswith('.stl'):
                mesh = trimesh.load_mesh(f'{path}\\{file}')
                meshes.append(mesh)
        return meshes
        
    # Figures and visualization for thesis
    def show_overgang_detection2D(self) -> None:
        # Only used for figure in thesis
        # Creates a 2d slice in xz-plane and visualizes the overhang detection algorithm
        
        dense = self.dense.copy()
        overhang_voxels = find_overhang_voxels(dense)
        overhang_mask = self.generate_overhang_mask(dense, t=10)
        
        # Pad arrays for visualization
        dense = np.pad(dense, 2, mode='constant', constant_values=0)
        overhang_voxels = np.pad(overhang_voxels, 2, mode='constant', constant_values=0)
        overhang_mask = np.pad(overhang_mask, 2, mode='constant', constant_values=0)
        
        # Get 2d slice in xz-plane
        dense_2d = dense[:, dense.shape[1]//2, :]
        overhang_2d = overhang_voxels[:, dense.shape[1]//2, :]        
        overhang_mask_2d = overhang_mask[:, dense.shape[1]//2, :]
        
        # # 2d slice in yz-plane
        # dense_2d = dense[dense.shape[0]//2, :, :]
        # overhang_2d = overhang_voxels[dense.shape[0]//2, :, :]
        # overhang_mask_2d = overhang_mask[dense.shape[0]//2, :, :]
    
        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        
        red_cmap = ListedColormap(['white', 'red'])
        green_cmap = ListedColormap(['white', 'green'])
        
        # bp1 = Rectangle((0,0), dense_2d.shape[0], 2, linewidth=1, edgecolor='black', facecolor='black')
        # bp2 = Rectangle((0,0), dense_2d.shape[0], 2, linewidth=1, edgecolor='black', facecolor='black')
        # bp3 = Rectangle((0,0), dense_2d.shape[0], 2, linewidth=1, edgecolor='black', facecolor='black')
        # bp4 = Rectangle((0,0), dense_2d.shape[0], 2, linewidth=1, edgecolor='black', facecolor='black')
        # bp5 = Rectangle((0,0), dense_2d.shape[0], 2, linewidth=1, edgecolor='black', facecolor='black')
        
        # Set gridlines
        for i in range(2):
            for j in range(3):
                ax[i,j].set_xticks(np.arange(-0.5, dense_2d.shape[0], 1))
                ax[i,j].set_yticks(np.arange(-0.5, dense_2d.shape[1], 1))
                ax[i,j].set_xticklabels([])
                ax[i,j].set_yticklabels([])
                ax[i,j].grid(color='w', linestyle='-', linewidth=0.5)

        
        
        # Dense voxel grid
        ax[0,0].imshow(dense_2d.T, cmap=green_cmap, origin='lower')
        # ax[0,0].add_patch(bp1)
        ax[0,0].set_title('a) Dense Voxel Grid')
        ax[0,0].set_aspect('equal')

        # Shifted up by 1 along z-axis overlaid on dense voxel grid
        img2_dense = np.zeros((dense_2d.shape[0], dense_2d.shape[1]), dtype=np.uint8)
        img2_dense[dense_2d] = 1
        img2_dense = np.pad(img2_dense, 1, mode='constant', constant_values=0)
        
        img2_shifted = np.roll(dense_2d, 1, axis=1)
        img2_shifted = np.pad(img2_shifted, 1, mode='constant', constant_values=0)
        
        ax[0,1].imshow(img2_dense.T, cmap=ListedColormap(['white', 'green']), origin='lower')
        ax[0,1].imshow(img2_shifted.T, cmap=ListedColormap([[1,1,1,0], [0, 0, 1, 0.4]]), origin='lower')
        # ax[0,1].add_patch(bp2)
        ax[0,1].set_title('b) Shifted up by 1')
        ax[0,1].set_aspect('equal')
        
        # Overhang voxels
        ax[0,2].imshow(overhang_2d.T, cmap=red_cmap, origin='lower')
        # ax[0,2].add_patch(bp3)
        ax[0,2].set_title('c) Overhanging voxels')
        ax[0,2].set_aspect('equal')
        
        # Overhanging voxels
        img3 = np.zeros((dense_2d.shape[0], dense_2d.shape[1]), dtype=np.uint8)
        img3[dense_2d] = 1
        img3[overhang_2d] = 2
        ax[1,0].imshow(img3.T, cmap=ListedColormap(['white', 'green', 'red']), origin='lower')
        # ax[1,0].add_patch(bp4)
        ax[1,0].set_title('d) Overhanging voxels and dense voxel grid')
        ax[1,0].set_aspect('equal')
        
        # Overhang mask after adding nonplanar height and flattening top
        img4 = np.zeros((dense_2d.shape[0], dense_2d.shape[1]), dtype=np.uint8)
        img4[dense_2d] = 1
        img4[overhang_mask_2d] = 2
        ax[1,1].imshow(img4.T, cmap=ListedColormap(['white', 'green', 'red']), origin='lower')
        # ax[1,1].add_patch(bp5)
        ax[1,1].set_title('e) After adding nonplanar height')
        ax[1,1].set_aspect('equal')
        
        ax[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
        fig.savefig('overhang_detection.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def show_padding(self) -> None:
        # Only used for figure in thesis
        # Creates a 2d slice in xz-plane and visualizes the padding algorithm
        
        pad = 3
        o_pad = 3
        
        dense = self.dense.copy()
        overhang_mask = self.generate_overhang_mask(dense, t=10)
        

        
        # Unpadded:
        dense_2d = dense[:, dense.shape[1]//2, :]
        overhang_2d = overhang_mask[:, dense.shape[1]//2, :]
        
        
        dense_2d = np.pad(dense_2d, pad, mode='constant', constant_values=0)
        overhang_2d = np.pad(overhang_2d, pad, mode='constant', constant_values=0)
        
        
        self.generate_scalar_field(alpha=45, t=10, pad=pad, o_pad=o_pad)
        scalar_field = self.scalar_field
        padded_dense = self.dense.copy()
        padded_overhang_mask = self.overhang_mask.copy()
        
        
        padded_dense_2d = padded_dense[:, padded_dense.shape[1]//2, :]
        padded_overhang_2d = padded_overhang_mask[:, padded_dense.shape[1]//2, :]
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        img = np.zeros_like(padded_dense_2d, dtype=np.uint8)
        
        img[dense_2d] = 1
        # overhang_2d[dense_2d] = np.nan
        img[overhang_2d] = 2
        img[padded_dense_2d & ~dense_2d] = 3
        img[padded_overhang_2d & ~overhang_2d] = 4
        
        img = np.pad(img, 2, mode='constant', constant_values=0)
        
        colormap = ListedColormap(['white', 'green', 'red', [0, 1, 0, 0.5], [1, 0.3, 0.3, 0.5]])
        
        ax.imshow(img.T, cmap=colormap, origin='lower')
        

        ax.set_xticks(np.arange(-0.5, dense_2d.shape[0]+1, 1))
        ax.set_yticks(np.arange(-0.5, dense_2d.shape[1]+1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='w', linestyle='-', linewidth=0.5)

        cbar0 = ax.figure.colorbar(cm.ScalarMappable(cmap=colormap), ax=ax, orientation='horizontal')
        # cbar0 = ax[0].figure.colorbar(cm.ScalarMappable(cmap=ListedColormap(['white', 'green', 'red'])), ax=ax[0], orientation='vertical', shrink=2/3)
        cbar0.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])  
        cbar0.set_ticklabels(['Empty', 'Supported', 'Overhang', 'Padded supported', 'Padded overhang'])


        plt.show()
        fig.savefig('padding.png', dpi=300, bbox_inches='tight')
        
    def show_distance_field(self) -> None:
        # Only used for figure in thesis
        # Creates a 2d slice in xz-plane and visualizes the distance field used in scalar field generation
        
    
        dense = self.dense.copy()
        overhang = self.generate_overhang_mask(dense, t=10)
        supported = ~overhang & dense
        
        
        
        distance_field = np.zeros_like(dense, dtype=np.float64)
        for z in range(dense.shape[2]):
            overhang_2d_slice = overhang[:, :, z]
            distance_field[:, :, z] = ndimage.distance_transform_edt(overhang_2d_slice)  # Compute 2d distance for every voxel/pixel in the 2d slice

        dense = np.pad(dense, 2, mode='constant', constant_values=0)
        overhang = np.pad(overhang, 2, mode='constant', constant_values=0)
        supported = np.pad(supported, 2, mode='constant', constant_values=0)
        distance_field = np.pad(distance_field, 2, mode='constant', constant_values=0)
        
            
        # Get 2D slice in xz-plane
        y_idx = dense.shape[1]//2
        dense_2d = dense[:, y_idx, :].T
        distance_field_2d = distance_field[:, y_idx, :].T
        overhang_2d_slice = overhang[:, y_idx, :].T
        dense_2d = dense[:, y_idx, :].T
        overhang_2d = overhang[:, y_idx, :].T
        supported_2d = supported[:, y_idx, :].T
        distance_field_2d = distance_field[:, y_idx, :].T
        distance_field_2d[dense_2d == 0] = np.nan
        
        
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 9))
        
        img0 = np.zeros((dense_2d.shape[0], dense_2d.shape[1], ), dtype=np.uint8)
        img0[supported_2d] = 1
        img0[overhang_2d] = 2
        
        ax[0].imshow(img0, cmap=ListedColormap(['white', 'green', 'red']), origin='lower')
        
        # ax[0].imshow(dense_2d, cmap='Greys', origin='lower')
        ax[0].set_title('a) Dense Voxel Grid')
        ax[0].set_xticks(np.arange(-0.5, dense_2d.shape[1], 1))
        ax[0].set_yticks(np.arange(-0.5, dense_2d.shape[0], 1))
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].grid(color='w', linestyle='-', linewidth=0.5)

        cbar0 = ax[0].figure.colorbar(cm.ScalarMappable(cmap=ListedColormap(['white', 'green', 'red'])), ax=ax[0], orientation='horizontal')
        # cbar0 = ax[0].figure.colorbar(cm.ScalarMappable(cmap=ListedColormap(['white', 'green', 'red'])), ax=ax[0], orientation='vertical', shrink=2/3)
        cbar0.set_ticks([1/6, 0.5, 5/6])  
        cbar0.set_ticklabels(['Empty', 'Supported', 'Overhang'])



                    
        ax[1].imshow(distance_field_2d, cmap='jet', origin='lower')

        
        ax[1].set_title('b) Distance Field')
        ax[1].set_xticks(np.arange(-0.5, dense_2d.shape[1], 1))
        ax[1].set_yticks(np.arange(-0.5, dense_2d.shape[0], 1))
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].grid(color='w', linestyle='-', linewidth=0.5)
        
        # Colorbar
        # ticks = np.arange(0, np.max(distance_field_2d[dense_2d])+1, 2)
        n_ticks = 11
        ticks = np.linspace(0, np.max(distance_field_2d[dense_2d]), n_ticks).astype(int)
        
        normalized_ticks = ticks/np.max(ticks)

        cbar1 = ax[1].figure.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax[1], orientation='horizontal')
        # cbar1 = ax[1].figure.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax[1], orientation='vertical', shrink=2/3)
        cbar1.set_ticks(normalized_ticks)
        cbar1.set_ticklabels(ticks)
        
        cbar1.set_label('Distance to nearest supported voxel [-]', fontsize=14, labelpad=10)
        
        fig.tight_layout()
        
        plt.show()
        fig.savefig('distance_field.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
         
    def show_scalar_field_generation(self) -> None:
        # Only used for figure in thesis
        # Creates a 2d slice in xz-plane and visualizes the scalar field generation algorithm
        
        dense = self.dense.copy()
        nonplanar_field = self.generate_scalar_field(alpha=45, t=10) * dense
        planar_field = self.planar_scalar_field * dense
        
        # Shift all values by 1 to avoid invisible voxels on the first layer in the visualization
        nonplanar_field[dense] += 1
        planar_field[dense] += 1
        
        nonplanar_field[~dense] = np.nan
        planar_field[~dense] = np.nan
        
        # planar_colormap = generate_random_cmap(np.max(planar_field))
        # scalar_colormap = generate_random_cmap(np.max(scalar_field))
        
        planar_colormap = cm.jet
        scalar_colormap = cm.jet
        
        
        planar_colormap.set_under(color='white')
        scalar_colormap.set_under(color='white')
        
        
        dense_2d = dense[:, dense.shape[1]//2, :].T
        nonplanar_field_2d = nonplanar_field[:, dense.shape[1]//2, :].T
        planar_field_2d = planar_field[:, dense.shape[1]//2, :].T
        
        # Pad for visualization
        dense_2d = np.pad(dense_2d, 1, mode='constant', constant_values=0)
        nonplanar_field_2d = np.pad(nonplanar_field_2d, 1, mode='constant', constant_values=np.nan)
        planar_field_2d = np.pad(planar_field_2d, 1, mode='constant', constant_values=np.nan)
        
        
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 9))
         
                    
        ax[0].imshow(dense_2d, cmap='Greys', origin='lower', alpha=0.5)
        ax[0].set_title('a) Dense Voxel Grid')
        ax[0].set_xticks(np.arange(-0.5, dense_2d.shape[1], 1))
        ax[0].set_yticks(np.arange(-0.5, dense_2d.shape[0], 1))
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].grid(color='w', linestyle='-', linewidth=0.5)

        # Planar field
        
        ax[1].imshow(planar_field_2d, cmap=planar_colormap, origin='lower')
        ax[1].set_title('b) Planar scalar field')
        ax[1].set_xticks(np.arange(-0.5, dense_2d.shape[1], 1))
        ax[1].set_yticks(np.arange(-0.5, dense_2d.shape[0], 1))
        ax[1].set_xticklabels([])
        ax[1].set_yticklabels([])
        ax[1].grid(color='w', linestyle='-', linewidth=0.5)
        
        
        # Scalar field
        ax[2].imshow(nonplanar_field_2d, cmap=scalar_colormap, origin='lower')
        ax[2].set_title('c) Non-planar scalar field')
        ax[2].set_xticks(np.arange(-0.5, dense_2d.shape[1], 1))
        ax[2].set_yticks(np.arange(-0.5, dense_2d.shape[0], 1))
        ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        ax[2].grid(color='w', linestyle='-', linewidth=0.5)

        
        ticks = np.arange(0, np.max(nonplanar_field_2d[dense_2d])+1, 1).astype(int)
        ticks_normalized = ticks/np.max(ticks)
        cb = ax[2].figure.colorbar(cm.ScalarMappable(cmap=scalar_colormap), ax=ax, orientation='horizontal')
        # cb = ax[2].figure.colorbar(cm.ScalarMappable(cmap=scalar_colormap), ax=ax, orientation='vertical', shrink=2/3)
        cb.set_ticks(ticks_normalized)
        cb.set_ticklabels(ticks)
        cb.set_label('Scalar field value', fontsize=14, labelpad=10)
        
        
        # fig.tight_layout()
        
        plt.show()
        fig.savefig('scalar_field_generation.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
  
    def show_smoothing(self, isovalue:int|float, n:int, c:float=1.0) -> None:

        scalar_field = self.generate_scalar_field(alpha=20, t=5, pad=3, o_pad=3)
        smoothed_mesh = self.generate_isosurface(scalar_field, isovalue=isovalue, n=n, c=c, trim=True, filter=False).as_open3d
        smoothed_mesh.compute_vertex_normals()
        smoothed_mesh.paint_uniform_color([0.5, 0.5, 0.8])

        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 800, height = 600)
        vis.add_geometry(smoothed_mesh)

        view_ctl = vis.get_view_control()
        view_ctl.set_front([0.0, 0.85, 0.54])
        view_ctl.set_lookat([0, 2.7, 4.4])
        
        view_ctl.set_up([0.0, -0.5, 0.8])
        view_ctl.set_zoom(0.5)
        view_ctl.translate(0, 15)
        
        render_options = vis.get_render_option()
        render_options.mesh_show_wireframe = True
        render_options.mesh_show_back_face = True
        
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        filename = 'mech_part_smoothed'
        vis.capture_screen_image(f'{filename}_n{n}_c{c}.png', do_render=True)
        vis.destroy_window()
    
    def show_winding_order(self) -> None:
        size = 3
        xs = np.arange(size)
        ys = np.arange(size)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6)) 
        

        
        # Add labels to the 4 points
        ax.text(0, 0, ' $v_1$ ', fontsize=15, ha='right', va='top')
        ax.text(1, 0, ' $v_2$ ', fontsize=15, ha='left', va='top')
        ax.text(1, 1, ' $v_3$ ', fontsize=15, ha='left', va='bottom')
        ax.text(0, 1, ' $v_4$ ', fontsize=15, ha='right', va='bottom')
        
        
        
        t1 = [
            [0, 1, 1, 0],
            [0, 0, 1, 0]
        ]
        
        t2 = [
            [0, 1, 0, 0],
            [0, 1, 1, 0]
        ]
        
        ax.plot(t1[0], t1[1], c='b', ls='--')        
        ax.plot(t2[0], t2[1], c='g', ls='--')
        
        # Fill the triangles
        ax.fill(t1[0], t1[1], color='b', alpha=0.2)
        ax.fill(t2[0], t2[1], color='g', alpha=0.2)
        
        # Add arrows to show winding order
        ax.arrow(0.15, 0.05, 0.6, 0.0, head_width=0.05, head_length=0.1, fc='b', ec='black')
        ax.arrow(0.95, 0.15, 0.0, 0.5, head_width=0.05, head_length=0.1, fc='b', ec='black')
        ax.arrow(0.88, 0.8, -0.6, -0.6, head_width=0.05, head_length=0.1, fc='b', ec='black')
        
        ax.arrow(0.15, 0.25, 0.59, 0.59, head_width=0.05, head_length=0.1, fc='g', ec='black')
        ax.arrow(0.85, 0.94, -0.6, 0.0, head_width=0.05, head_length=0.1, fc='g', ec='black')
        ax.arrow(0.075, 0.9, 0.0, -0.6, head_width=0.05, head_length=0.1, fc='g', ec='black')
        
        
        

        
        
        
        for x in xs:
            for y in ys:
                ax.scatter(x, y, color='black', s=20)
        
        
        
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.25, size-0.75)
        ax.set_ylim(-0.25, size-0.75)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.show()
        fig.savefig('grid_mesh_order.png', dpi=200, bbox_inches='tight')
        
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        size = 20
        
        xs = np.arange(size)
        ys = np.arange(size)
        
        
        
        
        def get_triangles(x, y):
            triangles = []
            v1 = [x, y]
            v2 = [x+1, y]
            v3 = [x+1, y+1]
            v4 = [x, y+1]
            triangles.append([v1, v2, v3]) 
            triangles.append([v1, v3, v4])
            return triangles
        
        
        
        
        
        triangles = []
        for x in xs[:-1]:
            for y in ys[:-1]:
                triangles.extend(get_triangles(x, y))
        
        for t in triangles:
            t = np.array(t)
            ax.plot(t[:, 0], t[:, 1], c='k', lw=1)
            ax.fill(t[:, 0], t[:, 1], color='k', alpha=0.2)
        ax.plot([0, 0], [0, size-1], c='k')
        
        X, Y = np.meshgrid(xs, ys)
        ax.scatter(X, Y, color='r', s=10)
        
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.show()
        fig.savefig('grid_mesh.png', dpi=200, bbox_inches='tight')
        
    def show_travel(self) -> None:
        fig, ax = plt.subplots( figsize=(18, 10))
        
        line0 = [
            [1, 3],
            [1, 2]
        ]
        
        
        ax.plot(line0[0], line0[1], c='k', ls='--', label='Original travel line', lw=2)
        
        safe_z = 2.5
        
        line1 = [
            [1, 1, 3, 3],
            [1, safe_z, safe_z, 2]
        ]
        
        
        ax.plot(line1[0], line1[1], c='g', label='Safe travel line', lw=2)
    
        ax.scatter(1, 1, c='r', label='Start point')
        ax.scatter(3, 2, c='b', label='End point')
    
        # Safe-z line
        ax.hlines(safe_z, 0.9, 3.1, color='gray', ls='dashdot', label='Safe z-level')
    
    
        ax.set_aspect('equal')
        
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
        
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        
        plt.show()
        fig.savefig('travel.png', dpi=300, bbox_inches='tight')

    def show_mesh_creation(self) -> None:
        scalar_field = self.generate_scalar_field(alpha=20, t=5, pad=3, o_pad=3)
        
        zs = interpolate_z_crossing(scalar_field, 70)
        
        
        # Flat mesh
        
        
        
        vertices, triangles = get_grid_mesh(zs)
        vertices = flatten_negative_vertices(vertices)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh_o3d = mesh.as_open3d
        mesh_o3d.compute_vertex_normals()
        
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width = 800, height = 600)
        vis.add_geometry(mesh_o3d)
        
        view_ctl = vis.get_view_control()
        view_ctl.set_front([0.0, 0.85, 0.54])
        view_ctl.set_lookat([0, 2.7, 4.4])
        
        view_ctl.set_up([0.0, -0.5, 0.8])
        view_ctl.set_zoom(0.5)
        view_ctl.translate(0, 15)
        
        render_options = vis.get_render_option()
        render_options.mesh_show_wireframe = True
        render_options.mesh_show_back_face = True
        
        vis.poll_events()
        vis.update_renderer()
        vis.run()
    
    def show_intersection(self) -> None:
        scalar_field = self.generate_scalar_field(alpha=20, t=5, pad=3, o_pad=3)
        isovalues = np.arange(72, np.max(scalar_field)+1, 5)
        for iso in isovalues:
            print(f'Isosurface: {iso}')
            zs = interpolate_z_crossing(scalar_field, iso)
            zs = smooth_z_coordinates(zs, 3)
            vertices, triangles = get_grid_mesh(zs)
            vertices = flatten_negative_vertices(vertices)
            vertices = vertices * self.pitch - self.transform
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            
            mesh = mesh_intersection(mesh, self.mesh, np_angle=20, max_area=0.5, filter=True, show=False)
            if mesh.is_empty:
                continue
            mesh_o3d = mesh.as_open3d
            mesh_o3d.compute_vertex_normals()
            
            vis = o3d.visualization.Visualizer()
            vis.create_window(width = 800, height = 600)
            vis.add_geometry(mesh_o3d)
            
            vis.run()
            vis.destroy_window()
     
    def show_trimming(self) -> None:
        scalar_field = self.generate_scalar_field(alpha=20, t=5, pad=3, o_pad=3)    
        isovalues = np.arange(1, np.max(scalar_field)+1, 1)
        isosurfaces = self.get_isosurfaces_mp(scalar_field, isovalues, n=3, c=1.0, export=False)
        
        isosurfaces = self.trim(isosurfaces, export=True)
        
        
        
        
        isosurfaces_o3d = [isosurf.as_open3d for isosurf in isosurfaces]
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        colors = generate_random_colors(len(isosurfaces))
        for i, surf in enumerate(isosurfaces_o3d):
            surf.compute_vertex_normals()
            surf.paint_uniform_color(colors[i])
            vis.add_geometry(surf)
        
        vis.run()
        

### HELPER FUNCTIONS ###

def get_grid_mesh(zs:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # zs: 2d array of z_heights
    # Returns the vertices and triangles of a mesh aligned in xy with the grid defined by zs
    xs = np.arange(zs.shape[0] - 1)
    ys = np.arange(zs.shape[1] - 1)
    vertices = []
    triangles = []
    for x in xs:
        for y in ys:
            v1 = [x, y, zs[x, y]]
            v2 = [x+1, y, zs[x+1, y]]
            v3 = [x+1, y+1, zs[x+1, y+1]]
            v4 = [x, y+1, zs[x, y+1]]
            vertices.extend([v1, v2, v3, v4])
            v1_index = len(vertices) - 4
            v2_index = len(vertices) - 3
            v3_index = len(vertices) - 2
            v4_index = len(vertices) - 1
            triangles.append([v1_index, v2_index, v3_index])
            triangles.append([v1_index, v3_index, v4_index])
    triangles = np.array(triangles)
    vertices = np.array(vertices)
    return vertices, triangles

def mesh_intersection(mesh:trimesh.Trimesh, cutting_mesh:trimesh.Trimesh, np_angle:float|int, max_area:float|None=None, filter:bool=True, show:bool=False) -> trimesh.Trimesh|None:
        # Computes the boolean intersection between mesh and cutting mesh, and and filters the result to remove unwanted triangles (if filter=True)
        tolerance = 1e-5
        # Visualization of mesh and cutting mesh
        if show:
            mesh_o3d = mesh.as_open3d
            cutting_mesh_o3d = cutting_mesh.as_open3d
            mesh_o3d.compute_vertex_normals()
            cutting_mesh_o3d.compute_vertex_normals()
            mesh_o3d.paint_uniform_color([0.5, 0.5, 0.5])
            cutting_mesh_o3d.paint_uniform_color([.8, .2, .2])
            o3d.visualization.draw_geometries([mesh_o3d, cutting_mesh_o3d])

        vertices_to_keep = mesh.vertices
        mesh_is_planar = is_planar(mesh)
        
        mesh = trimesh.boolean.intersection([mesh, cutting_mesh])
        # mesh = trimesh.boolean.intersection([mesh, cutting_mesh], check_volume=False, engine='blender')       
        # mesh = trimesh.boolean.intersection([mesh, cutting_mesh], check_volume=False, engine='manifold')   #  
            
        if mesh.is_empty:
            return mesh  
        
        # The trimesh.boolean.intersection method sometimes returns triangles that are inherited from the cutting, which we must remove manually (better alternative should be explored)
        if filter:
            keep_tree = KDTree(vertices_to_keep) 
            new_vertices = []
            new_faces = []
            vertex_dict = {} 

            for face in mesh.faces:
                new_face = []
                has_original_vertex = False
                for vertex_idx in face:
                    vertex = mesh.vertices[vertex_idx]

                    # Find the closest original vertex within toleranc
                    dist, closest_idx = keep_tree.query(vertex, k=1, distance_upper_bound=tolerance)

                    if dist < tolerance:
                        has_original_vertex = True
                        if closest_idx in vertex_dict:
                            new_face.append(vertex_dict[closest_idx])
                        else:
                            new_vertices.append(vertices_to_keep[closest_idx])
                            vertex_dict[closest_idx] = len(new_vertices) - 1
                            new_face.append(vertex_dict[closest_idx])
                    else:
                        new_vertices.append(vertex)
                        new_face.append(len(new_vertices) - 1)

                # Only add faces that have at least one original vertex
                if has_original_vertex and len(new_face) == 3:
                    new_faces.append(new_face)

            # Create a new mesh from the filtered data
            filtered_mesh = trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces))
            
            # Filter faces by angle: if it is planar, remove faces with normal not colinear with z-axis, if nonplanar remove faces with normal angles exceeding nonplanar_angle
            if mesh_is_planar:
                filtered_mesh = filter_by_normals(filtered_mesh, max_angle=0.1)
            else:
                filtered_mesh = filter_by_normals(filtered_mesh, max_angle=np_angle*1.1)
            mesh = filtered_mesh
    
        return mesh 

def filter_by_normals(mesh:trimesh.Trimesh, max_angle:float=45.) -> trimesh.Trimesh:
    # Removes faces where the angle between face normal and z-axis exceeds max_angle
    # max_angle: maximum angle in degrees (default = 45, hardware limitation from E3D belt nozzle)

    layer_normals = mesh.face_normals
    z_axis = np.array([0, 0, 1])
    normals_mask = np.arccos(np.dot(layer_normals, z_axis)) > np.radians(max_angle)
    mesh.update_faces(~normals_mask)
    if mesh.is_empty:
        # print('Empty mesh after normal filtering')
        return mesh
    return mesh
        
def get_neighbors(x:int, y:int, n:int) -> list[list[int]]:
    # Returns the indices of the n nearest 2d neighbors of the point (x, y)
    neighbors = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if i == 0 and j == 0:
                continue
            neighbors.append([x+i, y+j])
    return neighbors

def find_overhang_voxels(voxel_array:np.ndarray) -> np.ndarray:
    # Returns a boolean array where True (1) indicates overhang voxels
    padded_array = np.pad(voxel_array, pad_width=1, mode='constant', constant_values=False)
    shifted_up = np.roll(padded_array, 1, axis=2)  # Shift up by 1 along z-axis
    overhang_voxels = padded_array & ~shifted_up
    overhang_voxels = overhang_voxels[1:-1, 1:-1, 1:-1] # Remove padding 
    return overhang_voxels

def find_surface_voxels(voxel_array:np.ndarray) -> np.ndarray:
    padded_array = np.pad(voxel_array, pad_width=1, mode='constant', constant_values=False)
    left = padded_array[:-2, 1:-1, 1:-1]
    right = padded_array[2:, 1:-1, 1:-1]
    up = padded_array[1:-1, :-2, 1:-1]
    down = padded_array[1:-1, 2:, 1:-1]
    front = padded_array[1:-1, 1:-1, :-2]
    back = padded_array[1:-1, 1:-1, 2:]    
    surface_voxels = (voxel_array & (~left | ~right | ~up | ~down | ~front | ~back))
    
    return surface_voxels

def pad_surface(voxel_array:np.ndarray, method:str|None=None) -> np.ndarray:
    surface_voxels = find_surface_voxels(voxel_array)
    surface_indices = np.argwhere(surface_voxels)
    new_shape = np.array(voxel_array.shape) + 2
    if method == 'not_top': # Padding the voxel geometry with 1 layer of True (1) voxels in all directions except the positive z-axis   (x+, x-, y+, y-, z-)
        # new_shape = np.array(voxel_array.shape) + [2, 2, 2]
        padded_array = np.zeros(shape=new_shape, dtype=bool)
        for i, j, k in surface_indices:

            padded_array[i:i+3, j:j+3, k:k+2] = True
        
    if method == 'xy':  # Padding the voxel geometry with 1 layer of True (1) voxels in the xy-plane    (x+, x-, y+, y-)
        padded_array = np.zeros(shape=new_shape, dtype=bool)
        for i, j, k in surface_indices:
            padded_array[i:i+3, j:j+3, k+1:k+2] = True
    
    elif method == None:    # Default: Padding the voxel geometry with 1 layer of True (1) voxels in all directions (x+, x-, y+, y-, z+, z-)
        # new_shape = np.array(voxel_array.shape) + 2    
        padded_array = np.zeros(shape=new_shape, dtype=bool)
        for i, j, k in surface_indices:
            padded_array[i:i+3, j:j+3, k:k+3] = True
    
    return padded_array

def flatten_negative_vertices(vertices:np.ndarray) -> np.ndarray:
    # Sets the z-coordinate of negative vertices to the z-coordinate of the closest positive vertex
    if len(vertices) == 0:
        print('No vertices')
        return vertices
    
    if not np.any(vertices[:, 2] == -1):
        # print('No negative z-values')
        return vertices
    
    negative_mask = (vertices[:, 2] == -1)
    negative_indices = np.argwhere(negative_mask).flatten()  # Flatten to get a 1D array
    negative_vertices = vertices[negative_mask]
    positive_points = vertices[~negative_mask]
    if positive_points.size == 0:
        return vertices
    positive_tree = KDTree(positive_points[:, :2])

    for index, vertex in zip(negative_indices, negative_vertices):
        distance, closest_positive_idx = positive_tree.query(vertex[:2])    
    
        vertices[index, 2] = positive_points[closest_positive_idx, 2]
    return vertices

def smooth_z_coordinates(z_coordinates:np.ndarray, n:int=1, c:float=1.0) -> np.ndarray:
    smoothed_z_coordinates = np.copy(z_coordinates)
    for x in range(z_coordinates.shape[0]):
        for y in range(z_coordinates.shape[1]):
            
            if z_coordinates[x, y] < 0:
                continue
            neighbors = get_neighbors(x, y, n)
            neighbor_zs = []    # Interpolated z coordinates of neighboring cells, excluding negative values
            neighbor_distances = []
            for nx, ny in neighbors:
                if nx < 0 or nx >= z_coordinates.shape[0] or ny < 0 or ny >= z_coordinates.shape[1]:
                    continue
                z = z_coordinates[nx, ny]
                if z > 0:
                    neighbor_zs.append(z)
                    neighbor_distances.append((x-nx)**2 + (y-ny)**2) 
                else:
                    # If z < 0, no voxel is present, so we treat (nx, ny) as a ghost node and assign it the z-value of mirrored node relative to (x, y) - dz
                    dx = x - nx
                    dy = y - ny
                    x_mirrored = x + dx
                    y_mirrored = y + dy
                    if x_mirrored < 0 or x_mirrored >= z_coordinates.shape[0] or y_mirrored < 0 or y_mirrored >= z_coordinates.shape[1]:
                        continue
                    z_mirrored = z_coordinates[x+dx, y+dy]
                    
                    if z_mirrored > 0:
                        dz = z_coordinates[x, y] - z_mirrored
                        neighbor_zs.append(z_coordinates[x, y] + dz)
                        # neighbor_zs.append(z_coordinates[x, y] + dz)
                        neighbor_distances.append((x-nx)**2 + (y-ny)**2)

            if len(neighbor_zs) > 0:
                neighbor_distances = np.array(neighbor_distances)
                neighbor_distances = np.sqrt(neighbor_distances)
                # neighbor_distances = neighbor_distances / np.max(neighbor_distances)  # Normalize distances
                neighbor_zs = np.array(neighbor_zs)
                smoothed_z_coordinates[x, y] =  (1-c) * smoothed_z_coordinates[x, y] + (c) * np.average(neighbor_zs, weights=1/neighbor_distances)
    return smoothed_z_coordinates

def smooth_z_coordinates2(z_coordinates: np.ndarray, n: int = 1, c: float = 1.0) -> np.ndarray:
    smoothed_z_coordinates = np.copy(z_coordinates)
    shape = z_coordinates.shape

    for x in range(shape[0]):
        for y in range(shape[1]):
            if z_coordinates[x, y] < 0:
                continue

            neighbors = get_neighbors(x, y, n)
            neighbor_zs = []
            neighbor_distances = []

            for nx, ny in neighbors:
                if 0 <= nx < shape[0] and 0 <= ny < shape[1]:
                    z = z_coordinates[nx, ny]
                    if z > 0:
                        neighbor_zs.append(z)
                        neighbor_distances.append((x - nx) ** 2 + (y - ny) ** 2)
                    else:
                        dx, dy = x - nx, y - ny
                        x_mirrored, y_mirrored = x + dx, y + dy
                        if 0 <= x_mirrored < shape[0] and 0 <= y_mirrored < shape[1]:
                            z_mirrored = z_coordinates[x_mirrored, y_mirrored]
                            if z_mirrored > 0:
                                dz = z_coordinates[x, y] - z_mirrored
                                neighbor_zs.append(z_coordinates[x, y] + dz)
                                neighbor_distances.append((x - nx) ** 2 + (y - ny) ** 2)

            if neighbor_zs:
                neighbor_distances = np.sqrt(neighbor_distances)
                neighbor_weights = 1 / neighbor_distances
                smoothed_z_coordinates[x, y] = (1 - c) * smoothed_z_coordinates[x, y] + c * np.average(neighbor_zs, weights=neighbor_weights)
    
    return smoothed_z_coordinates
 
def is_planar(mesh:trimesh.Trimesh) -> bool:
    unique_zs = np.unique(np.round(mesh.vertices[:, 2], 5))
    if len(unique_zs) == 1:
        return True
    
    
@jit(nopython=False, parallel=True)
def interpolate_z_crossing(scalar_field:np.ndarray, isovalue:int|float) -> np.ndarray:
    z_heights = np.zeros(shape=(scalar_field.shape[:2]), dtype=np.float64)
    shape = scalar_field.shape
    # Iterate over all x, y points, find the voxels that contain the isosurface crossing and interpolate the exact z-height
    for x in range(shape[0]):
        for y in range(shape[1]):
            if x >= shape[0] or y >= shape[1]:
                continue

            for z in range(scalar_field.shape[2] - 1):
                val1 = scalar_field[x, y, z]
                val2 = scalar_field[x, y, z + 1]

                if val1 <= isovalue and val2 > isovalue and val1 > 0 and val2 > 0:
                    t = (isovalue - val1) / (val2 - val1)
                    z_height = z + t
                    z_heights[x, y] = z_height
                    break
            else:  
                z_heights[x, y] = -1
    return z_heights

def visualize_pointcloud(points:np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def generate_random_colors(num_colors:int) -> np.ndarray:
    colors = distinctipy.get_colors(num_colors)
    return colors
    
def generate_random_cmap(num_colors:int) -> ListedColormap:
    colors = generate_random_colors(num_colors)
    random_cmap = ListedColormap(colors, num_colors)
    return random_cmap


VERBOSE=True    # Enables printing of debug information

def main():   
    # path = 'meshes\\right_angle_overhang.stl'
    path = 'meshes\\internal_external.stl'
    # path = 'meshes\\mech_part.stl'
    
    


    layer_thickness_mm = 0.2
    pitch = .6 # Distance between voxels [mm]
    t = 5 # Vertical nonplanar height [mm]
    
    pad = 3 # Number of layers to pad the dense voxel grid with 
    o_pad = 3 # Number of layers to pad the overhang mask with (all directions except z+)
    
    n = 3  # Number of neighboring vertices to use when smoothing the interpolated isosurface
    c = 1 # Smoothing factor for the interpolation of the isosurface  (0 < c < 1) 0: no smoothing, 1: full smoothing (average of n neighbors)
    isovalue_offset = 0.0

    # Nonplanar layer angle
    alpha = 16.

    nplg = NPLayerGenerator(path, pitch=pitch, layer_thickness_mm=layer_thickness_mm)
    nplg.generate_scalar_field(alpha=alpha, t=t, pad=pad, o_pad=o_pad)

    crop = True
    if __name__ == "__main__":
        nplg.show_voxelgrid(nplg.overhang_mask, crop=crop)
        nplg.show_scalar_field(nplg.scalar_field, show_mesh=False, crop=crop)
    
    isovalues = np.arange(1+isovalue_offset, np.max(nplg.scalar_field)+1, 1.)
    isosurfaces = nplg.get_isosurfaces_mp(nplg.scalar_field, isovalues=isovalues, n=n, c=c, export=False)
    trimmed_meshes = nplg.trim(meshes = isosurfaces, export=True)
    
    if __name__ == "__main__":
        nplg.show_isosurfaces(trimmed_meshes, crop=crop, color='random', show_input_mesh=False)

if __name__ == "__main__":
   main()
