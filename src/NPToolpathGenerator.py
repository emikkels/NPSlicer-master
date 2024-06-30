import os
from datetime import datetime
import math
import numpy as np
import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt 

import trimesh

import shapely.plotting as splt
import shapely.geometry as sg
from shapely.ops import nearest_points

import networkx as nx
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator

from skspatial.measurement import area_signed

class ToolpathGenerator:
    def __init__(self, 
                mesh:trimesh.Trimesh, 
                layer_number:int,
                output_path:str, 
                layer_thickness:float, 
                infill_percentage:int|float, 
                nozzle_diameter:float, 
                perimeter_line_width:float,
                infill_line_width:float,
                n_perimeters:int=1, 
                infill_angle:int|float=0,
                infill_overlap_percentage:int|float=0,
                z_hop:float=0,
                safe_z:float|None=None,
                max_staydown_distance:float=0,
                max_edge_length:float=0.1,
                filament_diameter:float=1.75,
                prev_layer:trimesh.Trimesh|None=None,
                layer_fan_start:int=3,
                seam_position:list[float]|None=None,
                perimeter_orientation:str='cw', 
                last_layer:bool=False,
                translate_xy:list[float]=[0., 0.],
                ) -> None:
        
        # Parameters
        self.layer_number = layer_number
        self.output_path = output_path
        self.layer_thickness = layer_thickness
        self.infill_percentage = infill_percentage
        self.nozzle_diameter = nozzle_diameter
        self.perimeter_line_width = perimeter_line_width
        self.infill_line_width = infill_line_width
        self.n_perimeters = n_perimeters
        self.infill_angle = infill_angle
        self.infill_overlap_percentage = infill_overlap_percentage
        self.z_hop = z_hop
        self.safe_z = safe_z
        self.max_staydown_distance = max_staydown_distance
        self.max_edge_length = max_edge_length
        self.filament_diameter = filament_diameter
        self.prev_layer = prev_layer
        self.layer_fan_start = layer_fan_start
        self.seam_position = seam_position
        self.perimeter_orientation = perimeter_orientation
        self.is_last_layer = last_layer

        self.infill_line_spacing = self.infill_line_width * 100/self.infill_percentage

        if not isinstance(self.prev_layer, trimesh.Trimesh):
            self.prev_layer = None

        
        # Load input mesh and remove degenerate faces
        self.input_mesh = mesh.copy()
        self.mesh = mesh
        self.mesh.merge_vertices(digits_vertex=2)
        self.mesh.update_faces(self.mesh.nondegenerate_faces())
        self.mesh = self.mesh.simplify_quadric_decimation(len(self.mesh.faces) // 1.1)
        
        
        self.mesh.remove_unreferenced_vertices()
        self.mesh.update_faces(self.mesh.nondegenerate_faces())
        
        self.boundary_vertices = find_closed_contours(self.mesh, show=False)

        # Get boundary polygons from mesh edges
        self.create_boundary_polygons()
        
        self.set_boundary_start() 
        self.boundary_hierarchy = self.generate_boundary_hierarchy()    
        # self.show_boundary()
           
        self.perimeters = []    # Perimeters for the boundary contours, ordered from inner to outer
        self.infill_boundaries = [] # Boundaries for infill regions
        self.generate_perimeters_and_infill_boundary()
        self.connect_perimeters()
        # self.show_perimeters_and_infill_boundary()

        self.generate_infill_lines() 
        self.connect_infill_lines()
        # self.show_infill()
        
        # Combine perimeters and infill
        self.combine_toolpath2d()
        # self.show_combined_toolpath()

        self.project_toolpath_to_mesh()
        self.added_toolpaths = self.to_gcode(translate_xy=translate_xy)
    
    def create_boundary_polygons(self) -> None:
        boundary_polygons = [sg.Polygon(boundary) for boundary in self.boundary_vertices]
        for i, polygon in enumerate(boundary_polygons):
            if polygon.geom_type == 'MultiPolygon':
                boundary_polygons.pop(i)
                
        # Remove anything that is not a polygon
        for i, polygon in enumerate(boundary_polygons):
            if not isinstance(polygon, sg.Polygon):
                boundary_polygons.pop(i)
                    
        # Remove polygons with less than 3 points
        for i, polygon in enumerate(boundary_polygons):
            if len(polygon.exterior.coords) < 3:
                boundary_polygons.pop(i)
        
        # Remove polygons with area less than 5x nozzle orifice area
        for i, polygon in enumerate(boundary_polygons):
            if polygon.area < 1:
                boundary_polygons.pop(i)        
        
        #  Sort polygons by area in descending order
        sorted_polygons = sorted(boundary_polygons, key=lambda p: p.area, reverse=True)
        
        for polygon in sorted_polygons:
            for other_polygon in sorted_polygons:
                if polygon == other_polygon:
                    continue
                if polygon.exterior.intersects(other_polygon.exterior):
                    sorted_polygons.remove(other_polygon)
                    
        self.boundary_polygons = sorted_polygons          
    
    def set_boundary_start(self) -> None:
        # Set the start point of the contour to the point on the contour closest to the specified point
        if self.seam_position is None:
            return
            
        new_polygons = []
        for polygon in self.boundary_polygons:
            exterior_coords = np.array(polygon.exterior.coords)
            exterior_coords2D = exterior_coords[:, :2]
            
            polygon_tree = KDTree(exterior_coords2D)
            
            if self.is_nonplanar():
                z_max_index = np.argmax(exterior_coords[:, 2])
                z_max_pos = exterior_coords[z_max_index]
                z_max_2d = z_max_pos[:2]
                
                _, start_index = polygon_tree.query(z_max_2d)
            else:
                _, start_index = polygon_tree.query(np.array(self.seam_position))
            new_coords = np.concatenate((exterior_coords[start_index:], exterior_coords[:start_index]))
            new_polygon = sg.Polygon(new_coords)
            new_polygons.append(new_polygon)
            
        self.boundary_polygons = new_polygons
  
    def connect_cluster(self, clustered_points:np.ndarray, source_start:np.ndarray|None=None, closed:bool=True) -> np.ndarray:
        # Take a cluster of points and connect them to form closed contours
        cluster_graph = nx.Graph()
        cluster_tree = KDTree(clustered_points)
        
        if source_start is None:
            source_start_index = np.argmin(clustered_points[:, :2], axis=0)[0]
            source_start = clustered_points[source_start_index]

        else:
            points_2D = clustered_points[:, :2]
            tree_2D = KDTree(points_2D)
            _, source_start_index = tree_2D.query(source_start)
            source_start = clustered_points[source_start_index]

        # Connect the points to form a closed contour
        visited = np.zeros(len(clustered_points), dtype=bool) 
        k = min(clustered_points.shape[0], 4)    # Starting with 4 nearest neighbors, increase if no unvisited points are found
        start_index = source_start_index
        curr_index = start_index
        while not np.all(visited):
            visited[curr_index] = True  
            _, indices = cluster_tree.query(clustered_points[curr_index], k=4)  # Find the closest k points
            indices = indices[indices < visited.size]  # Filter out indices that are out of bounds
            unvisited_indices = indices[~visited[indices]]               # Filter out the points that have been visited
            while unvisited_indices.size == 0:
                k += 1
                if k > len(clustered_points):   # If k exceeds the number of points in the cluster, break the loop and connect the last point with the first point if closed=True
                    if closed:
                        cluster_graph.add_edge(curr_index, start_index)
                    break
                
                distances, indices = cluster_tree.query(clustered_points[curr_index], k=k)
                unvisited_indices = indices[~visited[indices]]
            if k > len(clustered_points):
                break
            next_index = unvisited_indices[0]
            k = 4  # Reset k to 4

            cluster_graph.add_edge(curr_index, next_index)
            curr_index = next_index
        if closed:
            # Extract closed contours from the graph, using the start point as the source
            edges = np.array(list(nx.eulerian_circuit(cluster_graph, source=source_start_index)))
        else:
            edges = np.array(list(nx.shortest_path(cluster_graph, source=source_start_index)))
        return clustered_points, edges

    def generate_boundary_hierarchy(self) -> dict:
        # Sort contours by area from largest to smallest
        sorted_polygons = self.boundary_polygons
        sorted_polygons.sort(key=lambda x: x.area, reverse=True)    # Sorted by area from largest to smallest
    
        # Buffer the polygons to remove self-intersections
        for i, polygon in enumerate(sorted_polygons):
            sorted_polygons[i] = polygon.buffer(0.001, quad_segs=16)
            
        if False:
            fig, ax = plt.subplots()
            for polygon in sorted_polygons:
                x, y = polygon.exterior.xy
                ax.plot(x, y, label=f'Boundary seed', c='k', linestyle='--', linewidth=0.5)
            ax.grid()
            ax.set_aspect('equal')
            plt.show()    
        
        # Iterate over the polygons and remove duplicates
        i = 0
        while i < len(sorted_polygons):
            j = i + 1
            while j < len(sorted_polygons):
                intersection_area = sorted_polygons[i].intersection(sorted_polygons[j]).area
                if intersection_area > 0.95*sorted_polygons[i].area:
                    del sorted_polygons[j]
                elif isinstance(sorted_polygons[i], sg.MultiPolygon):
                    if sorted_polygons[i].intersects(sorted_polygons[j]):
                        del sorted_polygons[j]
                    else:
                        j += 1
                elif isinstance(sorted_polygons[j], sg.MultiPolygon):
                    if sorted_polygons[j].intersects(sorted_polygons[i]):
                        del sorted_polygons[j]
                    else:
                        j += 1
                    
                elif sorted_polygons[i].exterior.intersects(sorted_polygons[j].exterior):
                    del sorted_polygons[j]
                else:
                    j += 1
            i += 1


        # Create a hierarchy of boundaries
        boundary_hierarchy = {polygon: {'level' : 0, 'parent' : None, 'children' : [], 'n_perimeters': self.n_perimeters} for polygon in sorted_polygons}
        for i, outer_polygon in enumerate(sorted_polygons):
            for inner_polygon in sorted_polygons[i+1:]:
                if outer_polygon.contains(inner_polygon):
                    boundary_hierarchy[inner_polygon]['level'] = boundary_hierarchy[outer_polygon]['level'] + 1
                    boundary_hierarchy[inner_polygon]['parent'] = outer_polygon
                    boundary_hierarchy[outer_polygon]['children'].append(inner_polygon)
                    break
        
        if False:  # Visualize the boundary hierarchy
            fig, ax = plt.subplots()
            for polygon, data in boundary_hierarchy.items():
                x, y = polygon.exterior.xy
                ax.plot(x, y, label=f'Boundary seed', c='k', linestyle='--', linewidth=0.5)
                if data['level'] is not None:
                    x, y = polygon.exterior.xy
                    ax.plot(x, y, label=f'Boundary {data["level"]}', c='r')
                    ax.scatter(x, y, c='k', s=5, label=f'Boundary {data["level"]}')
            ax.grid()
            ax.set_aspect('equal')
            plt.show()

        return boundary_hierarchy

    def adjust_perimeter_number(self) -> None:
        # Adjust the number of perimeters to avoid self-intersections
        hierarchy = self.boundary_hierarchy
        sorted_levels = sorted(set([data['level'] for data in hierarchy.values() if data['level'] is not None]))
        for level in sorted_levels:
            next_level = level + 1 if level is not None else 1
            level_polygons = [p for p, data in hierarchy.items() if data['level'] == level]
            next_level_polygons = [p for p, data in hierarchy.items() if data['level'] == next_level]
            for outer_polygon in level_polygons:
                distances = [outer_polygon.exterior.distance(inner_polygon) for inner_polygon in next_level_polygons]
                if distances:
                    min_distance = np.min(distances)
                    if min_distance < self.perimeter_line_width*hierarchy[outer_polygon]['n_perimeters'] + self.perimeter_line_width:
                        # Adjust the number of perimeters
                        hierarchy[outer_polygon]['n_perimeters'] = max(0, hierarchy[outer_polygon]['n_perimeters'] - 1)

                        for child_polygon in hierarchy[outer_polygon]['children']:
                            hierarchy[child_polygon]['n_perimeters'] = hierarchy[outer_polygon]['n_perimeters']

    def generate_perimeters_and_infill_boundary(self) -> None:
        for i in range(self.n_perimeters):
            self.adjust_perimeter_number()
        
        perimeters = {}
        infill_boundaries = {}
        # Generate perimeters for the boundary contours
        for contour, contour_data in self.boundary_hierarchy.items():
            level = contour_data['level']
            if level == None or level % 2 == 0:       # External contour
                offset_side = -1
            else:                                     # Internal contour
                offset_side = 1


            # Generate perimeters
            n_contours = contour_data['n_perimeters']   # Number of perimeters to generate for the current contour
            if n_contours == 0:
                continue

            for i in range(n_contours+1, 0, -1):
                if i == 1:  # Outermost perimeter
                    offset = self.perimeter_line_width / 2
                elif i == n_contours+1: # Infill boundary
                    offset = self.perimeter_line_width * n_contours * (1 - self.infill_overlap_percentage/100)
                else:
                    offset = self.perimeter_line_width*(i-0.5)

                offset_perimeter = contour.buffer(offset_side * offset, resolution=16).simplify(0.02, preserve_topology=False).segmentize(0.5)
                if offset_perimeter.is_empty :
                    continue
                offset_perimeters = []
                # Handle the case where offset polygon intersects itself and split it into multiple polygons 
                if offset_perimeter.geom_type == 'MultiPolygon':
                    offset_perimeters.extend(offset_perimeter.geoms)
                else:
                    offset_perimeters.append(offset_perimeter)

                for offset_perimeter in offset_perimeters:
                    # Ensure correct orientation
                    if self.perimeter_orientation == 'cw' and offset_perimeter.exterior.is_ccw:  # If the perimeter should be oriented clockwise and the contour is counter clockwise, reverse the contour
                        offset_perimeter = offset_perimeter.reverse()
                        # print('Reversed ccw to cw')
                    elif self.perimeter_orientation == 'ccw' and not offset_perimeter.exterior.is_ccw:  # If the perimeter should be oriented counter clockwise and the contour is clockwise, reverse the contour
                        offset_perimeter = offset_perimeter.reverse()
                        # print('Reversed cw to ccw' )
                
                    if i == n_contours+1:
                        infill_boundaries[offset_perimeter] = {'parent':contour, 'level': level}
                    else:
                        perimeters[offset_perimeter] = {'parent':contour,'level': level, 'perimeter_number': i}

        delete_perimeters = set()
        sorted_perimeters = sorted(perimeters.items(), key=lambda perimeter: perimeter[1]['perimeter_number'], reverse=True)
        # Check for intersection between perimeters
        for perimeter1, data1 in sorted_perimeters:
            if perimeter1 in delete_perimeters:
                continue
            for perimeter2, data2 in sorted_perimeters:
                if perimeter2 in delete_perimeters or perimeter1 == perimeter2:
                    continue
                    
                intersects = perimeter1.exterior.intersects(perimeter2.exterior)
                distance = perimeter1.exterior.distance(perimeter2.exterior)
                if intersects or distance < self.perimeter_line_width/2 * 0.9:
                    # Keep the lower level perimeter number
                    
                    
                    if data1['perimeter_number'] > data2['perimeter_number']:
                        delete_perimeters.add(perimeter1)
                    else:
                        delete_perimeters.add(perimeter2)
        
                    if False: # Draw the two perimeters fo
                        fig, ax = plt.subplots()
                        ax.set_aspect('equal')
                        
                        for perimeter, data in perimeters.items():
                            x, y = perimeter.exterior.xy
                            ax.plot(x, y, c='r', lw=1)
                        
                        x, y = perimeter1.exterior.xy
                        ax.plot(x, y, c='b', lw=2, label=f'Perimeter number: {data1["perimeter_number"]}, Level: {data1["level"]} ')
                        x, y = perimeter2.exterior.xy
                        ax.plot(x, y, c='g', lw=2, label=f'Perimeter number: {data["perimeter_number"]} Level: {data2["level"]}')
                        
                        ax.plot([], [], c='r', lw=1, label=f'Intersects: {intersects}, Distance: {np.round(distance, 3)}')
                        ax.legend()
                        
                        
                        plt.show()
                    
        for perimeter in delete_perimeters:
            del perimeters[perimeter]
                                    
        self.perimeters = perimeters
        self.infill_boundaries = infill_boundaries
        
    def connect_perimeters(self) -> None:
        # Group perimeters by parent boundary
        grouped_perimeters = {}
        for perimeter, data in self.perimeters.items():
            if data['parent'] not in grouped_perimeters:
                grouped_perimeters[data['parent']] = []
            grouped_perimeters[data['parent']].append({perimeter: data})

        connected_perimeter_groups = {}
        for parent, perimeter_group in grouped_perimeters.items():
            group_edges = []
            group_edge_types = []
            # Sort perimeter groups by perimeter_number    (1 is outermost/visible perimeter and should be last)
            sorted_perimeter_group = sorted(perimeter_group, key=lambda x: list(x.values())[0]['perimeter_number'], reverse=True)
            
            # Connect the perimeters within the same group
            for i, perimeter_data in enumerate(sorted_perimeter_group):
                
                if parent not in connected_perimeter_groups.keys():
                    connected_perimeter_groups[parent] = {}
                
                perimeter = list(perimeter_data.keys())[0]
                if perimeter.is_empty:
                    continue
                perimeter_points = np.array(perimeter.exterior.coords)
                # Connect pairs of points to form edges and add them to the group_edges list
                
                # Connect next perimeter to the previous one with a travel move
                if i > 0 and len(group_edges):
                    prev_endpoint = group_edges[-1][-1] # Last point of the previous perimeter
                    start = perimeter_points[0]
                    group_edges.append((prev_endpoint, start))
                    group_edge_types.append('travel')



                for p1, p2 in zip(perimeter_points[:-1], perimeter_points[1:]):
                    group_edges.append((p1, p2))
                group_edges.append((perimeter_points[-1], perimeter_points[0]))
                
                group_edge_types.extend(['perimeter' for _ in range(len(perimeter_points))])
            connected_perimeter_groups[parent]['edges'] = np.array(group_edges)
            connected_perimeter_groups[parent]['edge_types'] = np.array(group_edge_types)
        self.connected_perimeter_groups = connected_perimeter_groups
                
    def generate_infill_lines(self, mode:str='line', offset_xy:float=0.) -> None:
        if self.infill_percentage == 0 or self.infill_line_spacing == np.inf or len(self.infill_boundaries) == 0:
            self.piecewise_ordered_infill_lines = None
            self.grouped_infill_lines = None
            return 
        
        # Create a polygon representing each infill region 
        outer_polygons = []
        hole_polygons = []
        if len(self.infill_boundaries) == 0:
            self.piecewise_ordered_infill_lines = None
            self.grouped_infill_lines = None
            return None
        
        # If we have multiple levels of multiple levels (i.e  not just external contours), classify contours as external or hole
        if np.any([data['level'] > 0 for data in self.infill_boundaries.values()]):
            for boundary, boundary_data in self.infill_boundaries.items():
                x, y = boundary.exterior.xy

                # Determine if polygon is external or internal based on the level
                level = boundary_data['level']
                if level % 2 == 0:  # 0, 2, 4, 6, ... are external contours
                    is_external = True
                else:            # 1, 3, 5, 7, ... are internal contours (holes)
                    is_external = False
                if is_external:
                    outer_polygons.append(boundary)
                else:
                    hole_polygons.append(boundary)
        else:
            outer_polygons = list(self.infill_boundaries.keys())

            hole_polygons = []

        corresponding_holes = [[] for _ in outer_polygons]  # Corresponding holes for each outer polygon [[hole1, hole2, ...], [hole1, hole2, ...], ...]
        for hole in hole_polygons:
            for i, outer in enumerate(outer_polygons):
                point = sg.Point(hole.exterior.coords[0])
                if sg.Polygon(outer).contains(point):
                    corresponding_holes[i].append(hole)
                    break
        # If no corresponding hole is found, the hole contour has been offset past the outer contour, meaning that the infill region is not valid
        # In this case, we set the piecewise_ordered_infill_lines to None and return
        if len(hole_polygons) > 0:
            if all([len(holes) == 0 for holes in corresponding_holes]):
                self.piecewise_ordered_infill_lines = None
                self.grouped_infill_lines = None

                return None

        # Generate infill regions
        infill_regions = []
        for i, outer in enumerate(outer_polygons):
            if len(corresponding_holes[i]) == 0:
                infill_regions.append([outer])
                continue
            min_distance = np.inf
            for hole in corresponding_holes[i]:
                distance = outer.exterior.distance(hole.exterior)
                if distance < min_distance:
                    min_distance = distance
            if min_distance < 2*self.perimeter_line_width:
                # Set infill edges to None and return
                self.piecewise_ordered_infill_lines = None
                self.grouped_infill_lines = None
                return None
            polygon = outer.difference(corresponding_holes[i] if len(corresponding_holes[i]) > 0 else None)
            infill_regions.append(polygon)

        # infill_regions now define area to be filled with infill lines


        angle_rad = np.deg2rad(self.infill_angle)
    
        if mode == 'line':
            plane_normal = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
            plane_origin = np.array([offset_xy, offset_xy, 0])
            line_direction = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

            self.infill_normal = plane_normal
            self.infill_origin = plane_origin

            radius = 100  # 
            rs = np.arange(-radius, radius, self.infill_line_spacing)
            untrimmed_infill_lines = []
            infill_lines = []
            for r in rs:
                p1 = - line_direction * radius + (plane_origin[:2] + plane_normal[:2] * r)
                p2 = line_direction * radius + (plane_origin[:2] + plane_normal[:2] * r)
                line2d = sg.LineString([p1, p2])
                
                
                # Intersect with boundary polygons
                for polygon in infill_regions:
                    if polygon[0].intersects(line2d):
                        trimmed_line = polygon[0].intersection(line2d)
                        if trimmed_line.is_empty:
                            continue
                        if trimmed_line.length < self.infill_line_width/2:
                            continue
                        
                        if trimmed_line.geom_type == 'LineString':
                            infill_lines.append(trimmed_line)
                            untrimmed_infill_lines.append(line2d)
                            continue

                        if trimmed_line.geom_type == 'MultiLineString':
                            for l in trimmed_line.geoms:
                                infill_lines.append(l)
                                untrimmed_infill_lines.append(line2d)
                            continue
            
        # Resample the lines to have a close to uniform distance between points (necessary for 3d projection later)
        # for i, infill_line in enumerate(piecewise_ordered_infill_lines):
        resampled_ordered_infill_lines = []
        for i, line in enumerate(infill_lines):
            if line.length > 0:
                curr_dist = 0
                resampled_coords = []

                while curr_dist < line.length:
                    point = line.interpolate(curr_dist)
                    resampled_coords.append(point.coords[0])
                    curr_dist += np.min([self.max_edge_length, line.length - curr_dist])
                resampled_coords.append(line.interpolate(line.length).coords[0])
                resampled_ordered_infill_lines.append(np.array(resampled_coords))

        
        # Show resampling
        if False:
            fig, ax = plt.subplots(figsize=(16, 10))
            
            
            for line in resampled_ordered_infill_lines:
                x, y = line.T
                # ax.plot(x, y, c='r', lw=2)
                ax.scatter(x[1:-1], y[1:-1], c='g', s=5)      
            
            for line in infill_lines:
                x, y = line.xy
                ax.scatter(x, y, c='r', s=8)
            
            # Legend labels
            ax.plot([], [], c='r', lw=2, label='Start and end point')
            ax.plot([], [], c='g', lw=2, label='Resampled points')
            ax.legend(bbox_to_anchor=(1, 1), fontsize=14)
            
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            plt.show()
            fig.savefig('resampled_infill_lines.png', dpi=300)
        
        
        # Show trimming
        if False:
            # Visualize with plt
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.set_aspect('equal')

            # Plot infill regions
            for infill_boundary, data  in self.infill_boundaries.items():
                x, y = infill_boundary.exterior.xy
                ax.plot(x, y, c='b', ls='--')
                if data['level'] % 2 == 0:
                    ax.fill(x, y, c='b', alpha=0.1)
                else:
                    ax.fill(x, y, c='w', alpha=1)
            
            
            # Untrimmed infill lines
            for line in untrimmed_infill_lines:
                x, y = line.xy
                ax.plot(x, y, c='k', linestyle='--', lw=1.2)
                
            # Trimmed infill lines    
            for i, line in enumerate(resampled_ordered_infill_lines):
                x, y = line.T
                ax.plot(x, y, c='r', lw=3)
                # ax.scatter(x, y, c='k', s=5)
            
            
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            
            
            # Legend labels
            ax.plot([], [], c='k', lw=2, ls='--', label='Untrimmed infill lines')
            ax.plot([], [], c='r', lw=2,          label='Trimmed infill lines')
            ax.plot([], [], c='b', lw=2, ls='--', label='Infill boundary')
            ax.legend(bbox_to_anchor=(1, 1), fontsize=14)
            
            
            ax.set_xticks([])
            ax.set_yticks([])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            fig.savefig('infill_lines.png',  dpi=300)
            plt.show()

        # For each infill line, assign parent boundary
        grouped_infill_lines = {}
        grouped_infill_line_types = {}
        for i, line in enumerate(resampled_ordered_infill_lines):
            for boundary, data in self.infill_boundaries.items():
                line_sg = sg.LineString(line)
                line_midpoint = sg.Point(line_sg.interpolate(0.5, normalized=True))
                if boundary.contains(line_midpoint):
                    if data['parent'] not in grouped_infill_lines:
                        grouped_infill_lines[data['parent']] = []
                        grouped_infill_line_types[data['parent']] = []
                    grouped_infill_lines[data['parent']].append(line)
                    grouped_infill_line_types[data['parent']].append('infill')
                    break

        self.grouped_infill_lines = grouped_infill_lines
        self.grouped_infill_line_types = grouped_infill_line_types
     
    def connect_infill_lines(self) -> None:
        grouped_infill_lines = self.grouped_infill_lines
        connected_infill_groups = {}
        
        if grouped_infill_lines is None:
            self.connected_ordered_infill_edges = []
            self.infill_edge_types = line_types = []
            self.connected_infill_groups = None
            return
        
        for boundary, line_group in grouped_infill_lines.items():
            # Build a graph of where each line is a node, and each edge is a connection between adjacent lines. Then, we use tsp to find a path that connects all the lines
            
            if len(line_group) == 0:
                continue
            if len(line_group) > 1:
                
                line_graph = nx.Graph()
                for i, line in enumerate(line_group):
                    line_graph.add_node(i)
                    
                # Find the two lines with start/end points closest to the end/start point of the perimeters
                perimeter_start = self.connected_perimeter_groups[boundary]['edges'][0][0]  # Assuming that the first point of next layer perimeter is close to the current layer perimeter start
                perimeter_end = self.connected_perimeter_groups[boundary]['edges'][-1][-1]
                
                infill_start_points = np.array([line[0] for line in line_group])
                infill_end_points = np.array([line[-1] for line in line_group])
                
                start_distances = np.linalg.norm(infill_start_points - perimeter_start, axis=1)
                end_distances = np.linalg.norm(infill_end_points - perimeter_end, axis=1)
                
                node_distances = np.min([start_distances, end_distances], axis=0)
                start_index = np.argmin(node_distances)
                node_distances[start_index] = np.inf
                end_index = np.argmin(node_distances)

                # Add an edge between the start and end with a very large distance
                line_graph.add_edge(start_index, end_index, weight=1e6)
                
                # Iterate over all pairs of lines and add edges between them
                for i, line1 in enumerate(line_group):
                    for j, line2 in enumerate(line_group):
                        if i == j:
                            continue
                        if (i == start_index and j == end_index) or (j == start_index and i == end_index):
                            continue
                        
                        # Add an edge between the all infill lines with a large distance, so that the tsp algorithm can use them if no better options are available
                        line_graph.add_edge(i, j, weight=1000)

                        p0_i = line1[0]
                        p1_i = line1[-1]
                        p0_j = line2[0]
                        p1_j = line2[-1]
                        
                        d1 = np.linalg.norm(p0_i - p0_j)
                        d2 = np.linalg.norm(p0_i - p1_j)
                        d3 = np.linalg.norm(p1_i - p0_j)
                        d4 = np.linalg.norm(p1_i - p1_j)
                        distances = [d1, d2, d3, d4]
                        min_distance = np.min(distances)
                        line_graph.add_edge(i, j, weight=min_distance)
            
                # Find the shortest path that connects all the lines using tsp
                path = nx.algorithms.approximation.traveling_salesman_problem(line_graph.to_undirected(), 
                                                                            cycle=False, 
                                                                            nodes=line_graph.nodes, 
                                                                            weight='weight')
                ordered_lines = [line_group[i] for i in path]

                # Next: connect the lines by their endpoints
                if False:
                    fig, ax = plt.subplots()
                    ax.set_aspect('equal')
                    ax.grid(True)
                    for i, line in enumerate(ordered_lines):
                        x, y = line.T
                        ax.plot(x, y, c=cm.jet(i/len(ordered_lines)))
                        ax.scatter(x, y, c='k', s=5)
                    plt.show()
                    plt.close(fig)


                # Remove duplicates from ordered_lines, keeping only the first occurence
                unique_ordered_lines = []
                seen_lines = set()
                for line in ordered_lines:
                    line = tuple(map(tuple, line))
                    key = tuple(line)
                    if key not in seen_lines:
                        unique_ordered_lines.append(line)
                        seen_lines.add(key)
                
                # Connect the lines by their endpoints
                ordered_lines = unique_ordered_lines
            else:
                ordered_lines = [line_group]

            line_types = []
            connected_infill_lines = []
            for i, line in enumerate(ordered_lines):
                if i == 0:  # Add first line as is
                    for p1, p2 in zip(line[:-1], line[1:]):
                        connected_infill_lines.append((p1, p2))
                        line_types.append('infill')
                else: # Add subsequent lines with travel moves between them

                    prev_endpoint = connected_infill_lines[-1][1]

                    start = np.array(line[0])
                    end = np.array(line[-1])
                    
                    # If neither start x or y is close to prev_start, reverse the line if the distance from end to prev_end is less than start to prev_end
                    # if not (np.isclose(start[0], prev_endpoint[0]) or np.isclose(start[1], prev_endpoint[1])):
                    dist_start = np.linalg.norm(start - prev_endpoint)
                    dist_end = np.linalg.norm(end - prev_endpoint)
                    if dist_start > dist_end:
                        line = line[::-1]
                        start = np.array(line[0])
                        end = np.array(line[-1])

                    # Add travel move from previos endpoint to the start of the current line
                    connected_infill_lines.append((prev_endpoint, start))
                    line_types.append('travel')

                    # Add the line segments
                    for p1, p2 in zip(line[:-1], line[1:]):
                        connected_infill_lines.append((p1, p2))
                        line_types.append('infill')

            if boundary not in connected_infill_groups.keys():
                connected_infill_groups[boundary] = {}
            connected_infill_groups[boundary]['edges'] = np.array(connected_infill_lines)
            connected_infill_groups[boundary]['edge_types'] = np.array(line_types)
        
        self.connected_infill_groups = connected_infill_groups

    def combine_toolpath2d(self) -> None:
        
        infill_first = False        
        toolpath_edges = [] # List of tuples (start, end) representing the toolpath edges
        toolpath_types = [] # 'perimeter', 'infill' or 'travel'
    
        for i, boundary in enumerate(self.boundary_hierarchy):
            if boundary.area < self.nozzle_diameter**2:
                continue
            data = self.boundary_hierarchy[boundary]
                
            if data['level'] == 0:
                # Get perimeters and infill lines which are children of the same boundary
                
                if boundary in self.connected_perimeter_groups.keys():
                    perimeter_edges = self.connected_perimeter_groups[boundary]['edges']
                    perimeter_edge_types = self.connected_perimeter_groups[boundary]['edge_types']
    
                else:
                    perimeter_edges = None
                    perimeter_edge_types = None
                
                # Add perimeter groups that are contained within the 0-level boundary
                for b in self.connected_perimeter_groups.keys():
                    if b == boundary:
                        continue
                    if boundary.contains(b.exterior):
                        if perimeter_edges is not None:
                            # Add travel move
                            prev_end = perimeter_edges[-1][-1]
                            start = self.connected_perimeter_groups[b]['edges'][0][0]
                            travel_line = np.array((prev_end, start))
                            perimeter_edges = np.append(perimeter_edges, np.array([travel_line]), axis=0)
                            perimeter_edge_types = np.append(perimeter_edge_types, 'travel')
                        else:
                            perimeter_edges = np.empty((0, 2, 2))
                            perimeter_edge_types = np.empty(0)
                            
                        perimeter_edges = np.append(perimeter_edges, self.connected_perimeter_groups[b]['edges'], axis=0)
                        perimeter_edge_types = np.append(perimeter_edge_types, self.connected_perimeter_groups[b]['edge_types'])

                    
                
                if self.connected_infill_groups is not None and boundary in self.connected_infill_groups.keys():
                    infill_lines = self.connected_infill_groups[boundary]['edges']
                    infill_line_types = self.connected_infill_groups[boundary]['edge_types']
                else:
                    infill_lines = None
                    infill_line_types = None

                # Combine the perimeters and infill with same parent with travels in between
                combined_edges = []
                combined_types = []
                if infill_first:
                    if infill_lines is not None:
                        combined_edges.extend(infill_lines)
                        combined_types.extend(infill_line_types)
                    if perimeter_edges is not None:
                        if infill_lines is not None: # Add travel move
                            prev_end = infill_lines[-1][-1]
                            start = perimeter_edges[0][0]
                            combined_edges.append(np.array((prev_end, start)))
                            combined_types.append('travel')
                        combined_edges.extend(perimeter_edges)
                        combined_types.extend(perimeter_edge_types)
                else:
                    if perimeter_edges is not None:
                        combined_edges.extend(perimeter_edges)
                        combined_types.extend(perimeter_edge_types)
                    if infill_lines is not None and infill_lines.size > 0:
                        if perimeter_edges is not None: # Add travel move
                            prev_end = perimeter_edges[-1][-1]
                            start = infill_lines[0][0]
                            combined_edges.append(np.array((prev_end, start)))
                            combined_types.append('travel')
                        combined_edges.extend(infill_lines)
                        combined_types.extend(infill_line_types)
                if i > 0 and len(toolpath_edges) > 0 and len(combined_edges) > 0:
                    prev_end = toolpath_edges[-1][-1]
                    start = combined_edges[0][0]
                    toolpath_edges.append(np.array((prev_end, start)))
                    toolpath_types.append('travel')

                toolpath_edges.extend(combined_edges)
                toolpath_types.extend(combined_types)
                
                if False:
                    fig, ax = plt.subplots()
                    ax.set_aspect('equal')
                    ax.grid(True)
                    for i, edge in enumerate(combined_edges):
                        x, y = np.array(edge).T
                        if combined_types[i] == 'perimeter':
                            ax.plot(x, y, c='k')
                            ax.scatter(x, y, c='b', s=3)
                        elif combined_types[i] == 'travel':
                            ax.plot(x, y, c='g', linestyle='--')
                        else:
                            ax.plot(x, y, c='r', linestyle='--')
                    plt.show()
                    plt.close(fig)
        
        if len(toolpath_edges) > 0:
            self.combined_toolpath_edges = toolpath_edges
            self.combined_toolpath_types = toolpath_types
        else:
            self.combined_toolpath_edges = None
            self.combined_toolpath_types = None
              
    def project_toolpath_to_mesh(self) -> None:
        # Takes the edges in the 2D toolpath and projects them in z to the surface of the mesh
        # Also computes the extrusion distance for each edge
        # For travel moves, add additional moves at a higher z-value to avoid collisions with the mesh
        # safe_z = np.max(self.mesh.bounds[1, 2]  + self.layer_thickness) # Safe distance above the mesh to avoid collisions
                        
        edges, types = self.combined_toolpath_edges, self.combined_toolpath_types
        if edges is None:
            self.toolpath_lines = None
            self.toolpath_line_types = None
            self.extrusion_distances = None
            return


        # mesh_vertices = self.mesh.vertices
        mesh_vertices = self.input_mesh.vertices
        xs, ys = mesh_vertices[:, 0], mesh_vertices[:, 1]
        zs = mesh_vertices[:, 2]

        # z_interp = LinearNDInterpolator(np.array([xs, ys]).T, zs, rescale=True)
        z_interp = LinearNDInterpolator(np.array([xs, ys]).T, zs, rescale=False)

        edges_3D = []
        for edge in edges:
            edge_3D = []
            for point in edge:
                x, y = point
                z = z_interp(x, y)
                edge_3D.append([x, y, z])
            edges_3D.append(edge_3D) 
        edges_3D = np.array(edges_3D)


        if self.prev_layer is not None and self.is_nonplanar():
            _, closest_distances, _ = trimesh.proximity.closest_point(self.prev_layer, mesh_vertices)
            extrusion_height_interp = LinearNDInterpolator(mesh_vertices, closest_distances)

        else:
            extrusion_height_interp = None


        layer_thickness = self.layer_thickness
        projected_edges = []
        extrusion_distances = []
        for i, line in enumerate(edges_3D):
            line_type = types[i]
            start, end = line
            
            if line_type == 'travel':

                extrusion_distances.append(0)
                projected_edges.append([start, end])
                continue
            
            if self.prev_layer is not None:
                if isinstance(extrusion_height_interp, LinearNDInterpolator):
                    # Find distance from line segment to previous layer mesh using precomputed distance field                
                    start_dist = extrusion_height_interp(start)[0]
                    end_dist = extrusion_height_interp(end)[0]

                    if math.isnan(start_dist) and math.isnan(end_dist):
                        avg_dist = self.layer_thickness
                    elif math.isnan(start_dist):
                        avg_dist = end_dist
                    elif math.isnan(end_dist):
                        avg_dist = start_dist
                    else:
                        avg_dist = (start_dist + end_dist)/2
                        
                    layer_thickness = np.clip(avg_dist, 0.2*self.layer_thickness, self.layer_thickness)
                    if math.isnan(layer_thickness):
                        pass
                else:
                    layer_thickness = self.layer_thickness
            
        
            line_length2D = np.linalg.norm(start[:2] - end[:2])
            line_width = self.perimeter_line_width if line_type == 'perimeter' else self.infill_line_width
            extrusion_volume = (line_length2D * line_width) * layer_thickness    # Layer thickness adjusted for proximity to previous layer
            filament_area = np.pi*(self.filament_diameter/2)**2
            extrusion_distance = (extrusion_volume / filament_area)            
            
            extrusion_distances.append(extrusion_distance)
            projected_edges.append([start + np.array([0, 0, 0]), end])
            
        # Add additional travel moves if z-coordinates of start and end are different 
        new_lines = []
        new_types = []
        new_extrusion_distances = []
        last_empty = None
        for i, (line, line_type) in enumerate(zip(projected_edges, types)):
            if line[0].size == 0 or line[1].size == 0:
                if last_empty is None:
                    last_empty = i
                continue
            else:
                if last_empty is not None:
                    new_lines = np.concatenate((new_lines, (projected_edges[last_empty-1][1], line[0]) ))
                    new_types = np.append(new_types, 'travel')
                    last_empty = None
            
            # Compare z coordinates of previous and next point, add travel move via safe_z if different
            
            line = np.array(line).reshape(-1, 3)
            start = np.array(line[0])
            end = np.array(line[1])

            if i == 0:
                new_lines.append(line)
                new_types.append(line_type)
                new_extrusion_distances.append(extrusion_distances[i])
                continue
            
            if line_type == 'travel':
                prev_z = start[2]
                curr_z = end[2]
                distance = np.linalg.norm(start-end)
                if (curr_z != prev_z and distance > self.max_staydown_distance) or distance > self.max_staydown_distance:
                    p1 = np.concatenate((start[:2], [self.safe_z]))
                    p2 = np.concatenate((end[:2], [self.safe_z]))
                    
                    new_lines.append(np.array([start, start]))
                    new_types.append('travel')
                    new_extrusion_distances.append(-0.8)    # Retract

                    new_lines.append(np.array([start, p1]))
                    new_types.append('travel')
                    new_extrusion_distances.append(0)

                    new_lines.append(np.array([p1, p2]))
                    new_types.append('travel')
                    new_extrusion_distances.append(0)

                    new_lines.append(np.array([p2, end]))
                    new_types.append('travel')
                    new_extrusion_distances.append(0)

                    new_lines.append(np.array([end, end]))
                    new_types.append('travel')
                    new_extrusion_distances.append(0.8) # De-retract



                else:   # Add travel move with retraction, without traveling via safe_z
                    if distance > self.max_staydown_distance:
                        new_lines.append(np.array([start, start]))
                        new_types.append('travel')
                        new_extrusion_distances.append(-0.8)    # Retract

                        new_lines.append(line)
                        new_types.append(line_type)
                        new_extrusion_distances.append(0)

                        new_lines.append(np.array([end, end]))
                        new_types.append('travel')
                        new_extrusion_distances.append(0.8)
                    else:
                        new_lines.append(line)
                        new_types.append(line_type)
                        new_extrusion_distances.append(0)
            else:
                new_lines.append(line)
                new_types.append(line_type)
                new_extrusion_distances.append(extrusion_distances[i])

        self.toolpath_lines = np.array(new_lines)
        self.toolpath_line_types = np.array(new_types)
        self.extrusion_distances = np.array(new_extrusion_distances)

        
        # Visualize the toolpath
        if False:
            # o3d visualization
            toolpath_o3d = o3d.geometry.LineSet()
            points = []
            lines = []
            colors = []

            for i, line in enumerate(self.toolpath_lines):
                if len(line) != 2:
                    continue
                edge_tuples = list(map(list, line))
                points.extend(edge_tuples)  
                lines.append([2*i, 2*i+1])  
                
                # Assign colors based on linetype (red=perimeter, blue=infill, green=travel)
                if self.toolpath_line_types[i] == 'perimeter':
                    colors.append([1, 0, 0]) 
                elif self.toolpath_line_types[i] == 'infill':
                    colors.append([0, 0, 1])
                else:
                    colors.append([0, 1, 0])
            
            points = np.array(points)
            lines = np.array(lines)
            colors = np.array(colors)
            
            if points.size == 0:
                print(f'No toolpath generated for {self.output_path}')
                return
            
            points[:, 2] += 0.05

            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(points)
            points_o3d.paint_uniform_color([0, 0, 0])
        
            
            toolpath_o3d = o3d.geometry.LineSet()
            toolpath_o3d.points = o3d.utility.Vector3dVector(points)
            toolpath_o3d.lines = o3d.utility.Vector2iVector(lines)
            toolpath_o3d.colors = o3d.utility.Vector3dVector(colors)

            mesh_o3d = self.input_mesh.as_open3d
            mesh_o3d.compute_vertex_normals()
            


            vis = o3d.visualization.Visualizer()
            vis.create_window()
        
            vis.add_geometry(toolpath_o3d)
            vis.add_geometry(points_o3d)
            vis.add_geometry(mesh_o3d)
        

            vis.get_render_option().point_size = 3
            vis.run()
            vis.destroy_window()

    def is_nonplanar(self) -> bool:
        # Check if the mesh is non-planar by checking if the z-coordinates of the vertices are not all the same
        z_coords = self.mesh.vertices[:, 2]
        is_planar = np.allclose(z_coords, np.max(z_coords), rtol=0, atol=1e-3)
        return not is_planar

    def to_gcode(self, translate_xy:list[int]=[0, 0]) -> None:
        # Converts the projected toolpath to gcode and appends it to the output file path
        if self.toolpath_lines is None:
            print(f'No toolpath generated for layer {self.layer_number}')
            return False
        
        self.is_nonplanar = self.is_nonplanar()
        if self.is_nonplanar:
            print(f'Non-planar mesh detected for layer {self.layer_number}')
        else:
            print(f'Planar mesh detected for layer {self.layer_number}')
            

        
        extrusion_feedrate = 1500
        travel_feedrate = 10800
        retract_feedrate = 2500

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)


        # Create gcode file it doesn't exist and add starting gcode
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w') as f:
                # Print a record of the settings used to generate the gcode using f-strings
                f.write('; Generated by NPToolpathGenerator\n')
                f.write(f'; Layer thickness: {self.layer_thickness}\n')
                f.write(f'; Nozzle diameter: {self.nozzle_diameter}\n')
                f.write(f'; Filament diameter: {self.filament_diameter}\n')
                f.write(f'; Perimeter line width: {self.perimeter_line_width}\n')
                f.write(f'; Infill line width: {self.infill_line_width}\n')
                f.write(f'; Infill angle: {self.infill_angle}\n')
                f.write(f'; Infill line spacing: {self.infill_line_spacing}\n')
                f.write(f'; Layer fan start: {self.layer_fan_start}\n')
                f.write(f'; Max staydown distance: {self.max_staydown_distance}\n')
                f.write(f'; Seam position: {self.seam_position}\n')
                f.write(f'; Translate XY: {translate_xy}\n')

                
                # Starting gcode from PrusaSlicer MK3 config
                f.write('M201 X1000 Y1000 Z200 E5000 ; sets maximum accelerations, mm/sec^2\n')
                f.write('M203 X200 Y200 Z12 E120 ; sets maximum feedrates, mm / sec\n')
                f.write('M204 S1250 T1250 ; sets acceleration (S) and retract acceleration (R), mm/sec^2\n')
                f.write('M205 X8.00 Y8.00 Z0.40 E4.50 ; sets the jerk limits, mm/sec\n')
                f.write('M205 S0 T0 ; sets the minimum extruding and travel feed rate, mm/sec\n')
                f.write('G90 ; use absolute coordinates\n')
                f.write('M83 ; extruder relative mode\n')
                f.write('M104 S180 ; set extruder preheat temp\n')
                f.write('M106 ; Fan on\n')
                f.write('M140 S60 ; set bed temp\n')
                f.write('M190 S60 ; wait for bed temp\n')
                f.write('M104 S215 ; set extruder temp\n')

                f.write('G28 W ; home all without mesh bed level\n')
                f.write('G80 ; mesh bed leveling\n')
                f.write('M104 S215 ; set extruder temp\n')
                f.write('M109 S215 ; wait for extruder temp\n')
                
                # Purge line
                f.write('G1 Z0.2 F720\n')
                f.write('G1 Y-3 F1000 ; go outside print area\n')
                f.write('G92 E0\n')
                f.write('M107 ; Fan off\n')
                f.write('G1 X60 E9 F1000 ; intro line\n')
                f.write('G1 X100 E12.5 F1000 ; intro line\n')
                f.write('G92 E0.0\n')

                f.write('M221 S100 ; M221 - Set extrude factor override percentage\n')
                f.write('G21 ; set units to millimeters\n')
                f.write('G90 ; use absolute coordinates\n')
                f.write('M83 ; use relative distances for extrusion\n')
                f.write('M900 K0.04 ; Filament gcode LA 1.5\n')
                f.write('M900 K18 ; Filament gcode LA 1.0\n')
                f.write('M107 ; Fan off\n')

        
        # Convert the projected toolpath to gcode
        with open(self.output_path, 'a') as f:
            f.write(f'\n;Start layer {self.layer_number}\n')
            f.write(f';     Nonplanar layer\n' if self.is_nonplanar else f';     Planar layer\n')
            f.write(f'M117 Layer {self.layer_number}\n')


            # Set the extruder to 0
            f.write('G92 E0.0 ; Reset extruder distance\n')
            # f.write(f'G1 E-0.8 F{retract_feedrate}\n')

            # Turn on fan 
            if self.layer_number == self.layer_fan_start:
                f.write('M106 ; Fan on\n')

            for i, line in enumerate(self.toolpath_lines):
                start, end = np.around(line, 3) # Round to 3 decimal places
                start[:2] += translate_xy
                end[:2] += translate_xy

                start = np.around(start, 3)
                end = np.around(end, 3)

                line_type = self.toolpath_line_types[i]

                if i == 0: # Move to the start of the first line
                    f.write(f'G1 Z{np.around(self.safe_z, 3)} F{travel_feedrate}\n')
                    f.write(f'G1 X{start[0]} Y{start[1]} F{travel_feedrate}\n')
                    f.write(f'G1 Z{start[2]} F{travel_feedrate}\n')
                    f.write(f'G1 E0.8 F{retract_feedrate}\n')  # De-retraction
                    f.write(f'G92 E0.0\n')    # Reset extruder distance
                    
                extrusion_length = np.round(self.extrusion_distances[i], 5)
                if line_type == 'travel':
                    if round(abs(extrusion_length), 1) == 0.8:
                        f.write(f'G1 X{end[0]} Y{end[1]} Z{end[2]} E{extrusion_length} F{retract_feedrate} ; Retraction\n')
                    else:
                        f.write(f'G1 X{end[0]} Y{end[1]} Z{end[2]} F{travel_feedrate} ; Travel\n')
                else:
                    if extrusion_length == 0:
                        continue
                    if line_type == 'perimeter':
                        if self.is_nonplanar:
                            f.write(f'G1 X{end[0]} Y{end[1]} Z{end[2]} E{extrusion_length} F{extrusion_feedrate//2} ; {line_type.capitalize()}\n')
                        else:
                            f.write(f'G1 X{end[0]} Y{end[1]} Z{end[2]} E{extrusion_length} F{extrusion_feedrate} ; {line_type.capitalize()}\n')
                        
                    else:   # Infill
                        f.write(f'G1 X{end[0]} Y{end[1]} Z{end[2]} E{extrusion_length} F{extrusion_feedrate} ; {line_type.capitalize()}\n')
            
            f.write(f'G1 E-0.8 F{retract_feedrate}\n')  # Retraction
            f.write(f'G1 Z{np.around(self.safe_z, 3)} F{travel_feedrate}\n')
            f.write(f';End of layer {self.layer_number}\n')


            if self.is_last_layer:
                # Wipe (set relative mode, move to X2.0, Y2.0)
                f.write('G91 ; relative mode\n')
                f.write(f'G1 X2.0 Y2.0 E-0.4 F{travel_feedrate} ; wipe and retract\n')
                f.write(f'G01 E-0.1 F{travel_feedrate} ; retract some more\n')
                f.write('G90 ; absolute mode\n')
                
                # Turn off heaters and fan
                f.write('M104 S0 ; turn off extruder\n')
                f.write('M140 S0 ; turn off bed\n')
                f.write('M107 ; fan off\n')

                # Move up
                f.write(f'G1 Z{self.safe_z} F{travel_feedrate} ; move up\n')

                # Present print
                f.write(f'G1 Y200 F{travel_feedrate//2} ; present print\n')
                
                # Home x
                f.write('G28 X ; home x\n')
                
                # Turn off motors
                f.write('M84 ; disable motors\n')
                

        print(f'Toolpath for layer {self.layer_number} appended to {self.output_path}\n')

        if self.is_last_layer:
            print(f'Toolpath generation completed')
        
        return True

    ############# Visualization for thesis #############

    def show_perimeters_and_infill_boundary(self) -> None:
        # Visualization using matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_aspect('equal')
        
        for _, perimeter in enumerate(self.perimeters.items()):
            if perimeter[0].is_empty:
                continue

            if perimeter[1]['perimeter_number'] == 1:
                x, y = perimeter[0].exterior.xy
                ax.plot(x, y, linewidth=2, c='b')
                ax.scatter(x[0], y[0], c='k', s=10)
            else:
                
                x, y = perimeter[0].exterior.xy
                ax.plot(x, y, linewidth=2, c='g')
                ax.scatter(x[0], y[0], c='k', s=10)
                
        # Add toolpath boundaries
        for b in self.boundary_hierarchy.keys():
            x, y = b.exterior.xy
            ax.plot(x, y, c='k', linestyle='--', lw=1)
            # ax.scatter(x[0], y[0], c='k', s=10)
        
        # Add infill boundaries
        for i, infill_boundary in enumerate(self.infill_boundaries.items()):
            x, y = infill_boundary[0].exterior.xy
            ax.plot(x, y, c='r', ls='--', lw=1)
            
        # Add travel moves
        for g in self.connected_perimeter_groups.values():
            for edge, edge_type in zip(g['edges'], g['edge_types']):
                if edge_type == 'travel':
                    x, y = edge.T
                    ax.plot(x, y, c='gray', lw=1, ls=':')    
        
        # Fill bounded area
        for boundary, b_data in self.boundary_hierarchy.items():
            if b_data['level'] % 2 == 0:
                alpha = 0.1
                c = 'b'
            else:
                alpha = .9
                c = 'w'
            x, y = boundary.exterior.xy
            ax.fill(x, y, c=c, alpha=alpha)
        
        legend_elements = [
                           plt.Line2D([0], [0], color='k', lw=3, linestyle='--', label='Toolpath Boundary'),
                           plt.Line2D([0], [0], color='b', lw=3, label='External Perimeter'),
                           plt.Line2D([0], [0], color='g', lw=3, label='Internal perimeter'),
                           plt.Line2D([0], [0], color='r', lw=3, linestyle='--', label='Infill Boundary'),
                           plt.Line2D([0], [0], color='gray', lw=3, linestyle=':', label='Travel Move'),
        ]
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), fontsize='20')        
        
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 1))
        plt.show()
        fig.savefig(f'perimeters_{self.layer_number}.png', dpi=400, bbox_inches='tight')

    def show_boundary(self) -> None:
        # Visualize the toolpath boundary
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        for i, (boundary, b_data) in enumerate(self.boundary_hierarchy.items()):
            x, y = boundary.exterior.xy
            ax.plot(x, y, label=f'Boundary {i}', c='r')
            if b_data['level'] % 2 == 0:
                alpha = 0.5
                c = 'b'
            else:
                alpha = 1.0
                c = 'w'
            ax.fill(x, y, c=c, alpha=alpha)
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        
        
        fig.savefig(f'boundary_{self.layer_number}.png', dpi=300, bbox_inches='tight')
        plt.show()
      
    def show_infill(self) -> None:
        if self.connected_infill_groups is None:
            print('No infill generated')
            return
        if not np.any([len(group['edges']) > 0 for group in self.connected_infill_groups.values()]):
            print('No infill generated')
            return
        
        # Visualize the infill boundaries
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_aspect('equal')
        
        for group in self.connected_infill_groups.values():
            if group['edges'].size == 0:
                continue
            
            # Start and end points
            p_start = group['edges'][0, 0]
            p_end = group['edges'][-1, 1]
            ax.scatter(p_start[0], p_start[1], c='g', s=10)
            ax.scatter(p_end[0], p_end[1], c='b', s=10)
            
            # Infill and travel lines
            for edge, edge_type in zip(group['edges'], group['edge_types']):
                x, y = edge.T
                if edge_type == 'travel':
                    ax.plot(x, y, c='g', linestyle='--')
                if edge_type == 'infill':
                    ax.plot(x, y, c='r')
        
        
        # Legend labels
        ax.plot([], [], c='r', label='Infill')
        ax.plot([], [], c='g', linestyle='--', label='Travel')
        
        ax.legend(bbox_to_anchor=(1, 1), fontsize=18, loc='upper left') 
        
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.show()
        fig.savefig(f'infill_line_order.png', dpi=300 )

    def show_combined_toolpath(self) -> None:
        # Visualize the combined toolpath
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_aspect('equal')
        for i, edge in enumerate(self.combined_toolpath_edges):
            x, y = edge.T
            if self.combined_toolpath_types[i] == 'perimeter':
                ax.plot(x, y, c='b')
            elif self.combined_toolpath_types[i] == 'travel':
                ax.plot(x, y, c='g', linestyle='--')
            else:
                ax.plot(x, y, c='r', linestyle='--')
        
        # Legend labels
        ax.plot([], [], c='b', label='Perimeter')
        ax.plot([], [], c='r', linestyle='--', label='Infill')
        ax.plot([], [], c='g', linestyle='--', label='Travel')
        
        
        # Start and end points
        p_start = self.combined_toolpath_edges[0][0]
        p_end = self.combined_toolpath_edges[-1][1]
        ax.scatter(p_start[0], p_start[1], c='k', s=100,label='Start point', marker='v', zorder=10)
        ax.scatter(p_end[0], p_end[1], c='k', s=100, label='End point', marker='^', zorder=10)
        
        ax.legend(bbox_to_anchor=(1, 1), fontsize=18, loc='upper left')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                
        plt.show()
        fig.savefig(f'combined_toolpath_{self.layer_number}.png', dpi=300)

   

def generate_toolpaths(input_path:str, 
                        output_path:str,
                        layer_thickness:float,
                        infill_percentage:int|float,
                        nozzle_diameter:float,
                        perimeter_line_width:float,
                        infill_line_width:float,
                        n_perimeters:int,
                        alternate_direction:bool,
                        infill_overlap_percentage:float,
                        z_hop:float,
                        filament_diameter:float,
                        layer_fan_start:int,
                        seam_position:list[float],
                        translate_xy:list[float]=[0, 0]) -> None:
    # filename = input_path.split('\\')[-2]
    # output_path = input_path + '\\toolpaths.gcode'

    isosurfaces = load_isosurfaces(input_path)

    max_staydown_distance = 1.5 * nozzle_diameter/(infill_percentage/100) 
    min_edge_length = 0.5   # 

    safe_z = layer_thickness
    n_offset = 0    # Used to offset the layer number in the gcode file when a layer produces no toolpath
    for n, mesh in enumerate(isosurfaces):
        print(f'Generating toolpath for layer {n}')
        # Set infill to alternate between 45 and -45 degrees
        
        if n % 2 == 0:
            infill_angle = 0
            if alternate_direction:
                perimeter_orientation = 'cw'
            else:
                perimeter_orientation = 'ccw'
        else:
            infill_angle = 90
            if alternate_direction:
                perimeter_orientation = 'ccw'
            else:
                perimeter_orientation = 'cw'

        
        safe_z = max(safe_z, mesh.bounds[1, 2] + layer_thickness)     # Keep a record of the highest z-value in each layer to avoid collisions when traveling
        
        tg = ToolpathGenerator(mesh, 
                        n-n_offset, 
                        output_path, 
                        layer_thickness, 
                        infill_percentage, 
                        nozzle_diameter,
                        perimeter_line_width,
                        infill_line_width, 
                        n_perimeters=n_perimeters,
                        infill_angle=infill_angle,
                        infill_overlap_percentage=infill_overlap_percentage,
                        z_hop=z_hop,
                        safe_z=safe_z,
                        max_staydown_distance=max_staydown_distance,
                        max_edge_length=min_edge_length,
                        filament_diameter=filament_diameter,
                        prev_layer = isosurfaces[n-1] if n > 0 else None,    
                        layer_fan_start = layer_fan_start,
                        seam_position = seam_position,
                        perimeter_orientation=perimeter_orientation,
                        last_layer=False if n < len(isosurfaces)-1 else True,
                        translate_xy=translate_xy)
                    
        if not tg.added_toolpaths:
            n_offset += 1
      
def load_isosurfaces(path:str) -> list[trimesh.Trimesh]:
    # Loads all stl files in the input directory and returns a list of trimesh objects sorted by isovalue/layer number
    files = os.listdir(path)   # Get all stl files in the input directory
    files = [f'{path}\\' + f for f in files if f.endswith('.stl')]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))    # Sort the files based on the isovalue in the filename
    meshes = [trimesh.load(file) for file in files]
    return meshes
    
def find_contour(start_vertex:np.ndarray, edge_dict:dict) -> list[np.ndarray]:
    # Helper function to find a single closed contour
    contour = [start_vertex]
    current_vertex = start_vertex
    while True:
        possible_vertices = edge_dict[current_vertex]
        next_vertex = None
        for vertex in possible_vertices:
            if vertex != current_vertex and vertex not in contour:
                next_vertex = vertex
                break
        if next_vertex is None or next_vertex == contour[0]:
            break
        contour.append(next_vertex)
        current_vertex = next_vertex
    return contour 

def find_closed_contours(mesh:trimesh.Trimesh, show=False) -> list[np.ndarray]:
    # Find closed contours in a mesh by grouping boundary edges
    boundary_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)] # Free edges/boundary edges: https://github.com/mikedh/trimesh/issues/1060
    boundary_verticex_indices = np.unique(boundary_edges.flatten())

    vertex_edges = {}  # Dictionary to group edges by vertex
    for edge in boundary_edges:
        for vertex_idx in edge:
            if vertex_idx not in vertex_edges:
                vertex_edges[vertex_idx] = []
            vertex_edges[vertex_idx].extend(edge)

    contours = []
    while boundary_verticex_indices.size > 0:
        contour = find_contour(boundary_verticex_indices[0], vertex_edges)
        contours.append(contour)
        boundary_verticex_indices = np.array([v_idx for v_idx in boundary_verticex_indices if v_idx not in contour])
    closed_contours = [[mesh.vertices[idx] for idx in contour] for contour in contours]
    closed_contours = [np.array(contour + [contour[0]]) for contour in closed_contours]

    # Visualize the closed contours
    if show:
        if len(closed_contours) == 1:
            vis_contour_segments = [closed_contours[0]]
        else:
            vis_contour_segments = closed_contours
        
        if True:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            mesh_o3d = mesh.as_open3d
            mesh_o3d.compute_vertex_normals()
            vis.add_geometry(mesh_o3d)
        
            for contour in vis_contour_segments:
                color = np.random.rand(3)
                
                vertices = o3d.geometry.PointCloud()
                vertices.points = o3d.utility.Vector3dVector(contour)
                vertices.paint_uniform_color(color)
                
                edges = [[i, i+1] for i in range(len(contour)-1)]
                lines = o3d.geometry.LineSet()
                lines.points = vertices.points
                lines.lines = o3d.utility.Vector2iVector(edges)
                lines.colors = o3d.utility.Vector3dVector([color for _ in range(len(edges))])
                
                vis.add_geometry(lines)  
                vis.add_geometry(vertices)  
            
            vis.run()
        
    
    return closed_contours

def is_clockwise(points2d:np.ndarray) -> bool:
    # Check if a set of 2D points are ordered in a clockwise or counter-clockwise orientation using the shoelace formula
    area = area_signed(points2d)    # Signed area from shoelace formula
    return area > 0  # True if area is positive(cw), False if area is negative(ccw)




def main() -> None:
    layer_numbers = np.arange(0, 600, 1)
    
    
    # input_dir = 'output\\right_angle_overhang\p1'
    # input_dir = 'output\\right_angle_overhang\p0.5'
    # input_dir = 'output\\right_angle_overhang\p0.25'

    
    # input_dir = 'output\\internal_external\p1'
    input_dir = 'output\\internal_external\p0.6'
    # input_dir = 'output\\internal_external\p0.5'
    # input_dir = 'output\\internal_external\p0.25'
    # input_dir = 'output\\internal_external\p0.2'

    # input_dir = 'output\\mech_part\p1'
    # input_dir = 'output\\mech_part\p0.5'
    # input_dir = 'output\\mech_part\p0.25'

    filename = input_dir.split('\\')[-2]
    output_path = input_dir + f'\\{filename}_{datetime.now().strftime("%d.%m.%y_%H.%M")}.gcode'  # Output path for the toolpath, will be saved as a .gcode file
    print(output_path)
    # Check if output_path exists and delete it if it doesq
    if os.path.exists(output_path):
        os.remove(output_path)
    
    files = os.listdir(input_dir)   # Get all stl files in the input directory
    files = [f'{input_dir}\\' + f for f in files if f.endswith('.stl')]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))    # Sort the files based on the layer number in the filename
    
    meshes = [trimesh.load(file) for file in files]

    layer_thickness = 0.2
    infill_percentage = 20
    nozzle_diameter = 0.6
    perimeter_line_width = 0.45
    infill_line_width = 0.6
    n_perimeters = 2
    infill_overlap_percentage = 0
    z_hop = layer_thickness
       
    max_staydown_distance = 1.5 * nozzle_diameter/(infill_percentage/100) 
    min_edge_length = 0.5   # mm
    filament_diameter = 1.75    # mm
    layer_fan_start = 3
    perimeter_start_xy = [-30, -15]
    translate_xy = [100, 100]

    safe_z = 0.0    

    for n, mesh in enumerate(meshes):
        if n not in layer_numbers:
            continue
        print(f'Generating toolpath for layer {n}')
        # Set infill to alternate between 45 and -45 degrees
        if n % 2 == 0:
            infill_angle = 0
            perimeter_orientation = 'cw'
        else:
            infill_angle = 90
            perimeter_orientation = 'ccw'
        
        safe_z = max(safe_z, mesh.bounds[1, 2] + layer_thickness)     # Keep a record of the highest z-value in each layer to avoid collisions when traveling
        
        ToolpathGenerator(mesh, 
                        n, 
                        output_path, 
                        layer_thickness, 
                        infill_percentage, 
                        nozzle_diameter,
                        perimeter_line_width,
                        infill_line_width, 
                        n_perimeters=n_perimeters,
                        infill_angle=infill_angle,
                        infill_overlap_percentage=infill_overlap_percentage,
                        z_hop=z_hop,
                        safe_z=safe_z,
                        max_staydown_distance=max_staydown_distance,
                        max_edge_length=min_edge_length,
                        filament_diameter=filament_diameter,
                        prev_layer = meshes[n-1] if n > 0 else None,    
                        layer_fan_start = layer_fan_start,
                        seam_position = perimeter_start_xy,
                        perimeter_orientation=perimeter_orientation,
                        last_layer=False if n < len(files)-1 else True,
                        translate_xy=translate_xy
                    )
    

    
if __name__ == "__main__":
    main()
