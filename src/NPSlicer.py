import os
import copy
import re
from datetime import datetime
import threading
import queue

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui # type: ignore
import open3d.visualization.rendering as rendering # type: ignore

import NPLayerGenerator
from NPToolpathGenerator import generate_toolpaths

root_directory = os.path.dirname(os.path.abspath(__file__))
root_data_directory = os.path.join(root_directory, 'data')
os.makedirs(root_data_directory, exist_ok=True)

class NPSlicer():
    def __init__(self):
        self.build_volume = [200, 200, 200]
        self.input_path = None
        self.input_mesh = None         
        self.input_mesh_filename = None 
        self.data_directory = None
        self.output_directory = None
        self.results_queue = None # Queue to store results from worker threads
        self.isosurfaces = []
        self.geometry_dict = {}
        self.geometry_visibility = {}
        self.show_wireframe = False
        
        self.T = np.eye(4)
        self.scale = np.array([1, 1, 1])
        
        self.pitch = 1.0
        self.angle = 20.0
        self.overhang_thickness = 5.0
        self.layer_thickness = 0.2
        self.voxel_pad = 3
        self.overhang_pad = 3
        self.smoothing_neighbors = 3
        self.smoothing_neighbors = 3
        self.smoothing_coeff = 1.0
        
        
        
        self.setup_materials()


        # Create the window
        self.window = gui.Application.instance.create_window('NPSlicer', 1920, 1080)
        # self.window.show_menu(True)
        # self.window.size_to_fit()
        self.window.set_on_key(self._on_key)
        self.setup_widget()


        self._on_toggle_build_volume(True)
        self.buildplate_bbox = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [self.build_volume[0], self.build_volume[1], 1])
        self.scene_widget.setup_camera(60, self.buildplate_bbox, [0, 0, 1])

    def setup_materials(self):

        # Material used for meshes and voxels
        self.material = rendering.MaterialRecord()        
        self.material.shader = 'defaultLitTransparency'
        # self.material.shader = 'defaultLit'
        # self.material.shader = 'defaultLitSSR'
        # self.material.shader = 'pbr'
        # self.material.shader = 'depth'
        
        self.material.base_color = [0.7, 0.7, 0.7, 1.0]
        self.material.base_roughness = 0.8  
        self.material.base_reflectance = 0.4 
        self.material.base_metallic = 0.0    
        # self.material.base_roughness = 0.0
        # self.material.base_reflectance = 0.0
        # self.material.base_clearcoat = 0.0
        # self.material.thickness = 1.0
        # self.material.transmission = 1.0
        # self.material.absorption_distance = 1
        # self.material.absorption_color = [0.5, 0.5, 0.5]
        

        # Line material used for wireframe and toolpath visualization
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = 'unlitLine'
        self.line_material.base_color = [1, 1, 1, 1.0]
        self.line_material.line_width = 1.0
        
    def setup_widget(self):

        # 3D widget
        #region widget setup
        self.scene_widget = gui.SceneWidget()
        # self.scene_widget.center_of_rotation = [self.build_volume[0]/2, self.build_volume[1]/2, 0]
        
        self.scene_widget.enable_scene_caching(True)
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (-1, -1, -1))
        self.scene_widget.scene.scene.set_sun_light([0, 1, -1], [1, 1, 1], 10000)
        self.scene_widget.scene.scene.enable_sun_light(True)
        self.scene_widget.scene.scene.enable_indirect_light(True)
        
        # self.scene_widget.scene.scene.enable_indirect_light(True)
        # self.scene_widget.scene.scene.add_point_light(
        #     name='light1',
        #     color=np.array([1, 1, 1]),
        #     position=np.array([self.build_volume[0]/2, self.build_volume[1]/2, self.build_volume[2]]),
        #     intensity=1.0,
        #     falloff=0.0,
        #     cast_shadows=False)
        # self.scene_widget.scene.scene.add_point_light(
        #     name='light2',
        #     color=np.array([1, 1, 1]),
        #     position=np.array([0, -self.build_volume[1]/2, self.build_volume[2]/2]),
        #     intensity=1.0,
        #     falloff=0.0,
        #     cast_shadows=False)
        
        self.window.add_child(self.scene_widget)

        # self.scene_widget.scene.scene.add_point_light([0, 0, 0], [1, 1, 1])
        # self.scene_widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.6, -0.6, -0.6))
        # self.scene_widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.6, 0.6, 0.6))
        # self.scene_widget.scene.set_lighting(rendering.Open3DScene.LightingProfile.MED_SHADOWS, (0.6, 0.6, 0.6))
        ground_plane = o3d.visualization.rendering.Scene.GroundPlane.XY
        self.scene_widget.scene.show_ground_plane(True, ground_plane)
        

        # Settings
        self.settings_vert = gui.CollapsableVert('Setting', 1, gui.Margins(5))
        self.settings_vert.background_color = gui.Color(0.0, 0.2, 0.05, 0.8)
        self.settings_vert.preferred_width = 500
        self.window.add_child(self.settings_vert)
        #endregion widget setup
        

        # Set up the verts
        self.machine_settings = gui.CollapsableVert('Machine', 1, gui.Margins(20))
        self.view = gui.CollapsableVert('View', 1, gui.Margins(20))
        self.load = gui.CollapsableVert('Load', 1, gui.Margins(20))
        self.transform = gui.CollapsableVert('Transform', 5, gui.Margins(20))
        self.layer_generation = gui.CollapsableVert('Layer Generation', 1, gui.Margins(20))
        self.toolpath_generation = gui.CollapsableVert('Toolpath Generation', 1, gui.Margins(20))
        
        self.machine_settings.set_is_open(False)
        self.view.set_is_open(False)
        self.load.set_is_open(True)
        self.transform.set_is_open(False)
        self.layer_generation.set_is_open(False)
        self.toolpath_generation.set_is_open(False)
        
        self.settings_vert.add_child(self.machine_settings)
        self.settings_vert.add_child(self.view)
        self.settings_vert.add_child(self.load)
        self.settings_vert.add_child(self.transform)
        self.settings_vert.add_child(self.layer_generation)
        self.settings_vert.add_child(self.toolpath_generation)


       
        
        ######################## Build volume settings
        #region build volume settings
        build_volume_panel = gui.CollapsableVert('Build Volume', 1, gui.Margins(10))
        build_volume_panel.set_is_open(False)
        self.machine_settings.add_child(build_volume_panel)
    
        build_volume_row = gui.Vert()
        build_volume_panel.add_child(build_volume_row)
        build_volume_label = gui.Label('Set build volume [x,y,z] mm')
        build_volume_row.add_child(build_volume_label)
        self.build_volume_setting = gui.VectorEdit()
        self.build_volume_setting.vector_value = [200, 200, 200]
        self.build_volume_setting.set_on_value_changed(self._on_build_volume_changed)
        build_volume_row.add_child(self.build_volume_setting)
        build_volume_row.add_fixed(5)
        #endregion build volume settings

        
        #################### View orientation buttons
        #region view orientation buttons
        view_orientation_panel = gui.CollapsableVert('View orientation', 5, gui.Margins(5))
        self.view.add_child(view_orientation_panel)
        view_orientation_vgrid = gui.VGrid(5, 5)
        view_orientation_panel.add_child(view_orientation_vgrid)
        
        # X+ view
        x_plus_button = gui.Button('X+')
        x_plus_button.set_on_clicked(self._on_view_x_plus)
        view_orientation_vgrid.add_child(x_plus_button)
        
        # Y+ view
        y_plus_button = gui.Button('Y+')
        y_plus_button.set_on_clicked(self._on_view_y_plus)
        view_orientation_vgrid.add_child(y_plus_button)
        
         # Z+ view
        z_plus_button = gui.Button('Z+')
        z_plus_button.set_on_clicked(self._on_view_z_plus)
        view_orientation_vgrid.add_child(z_plus_button)
        
        # Iso 1 view
        iso1_button = gui.Button('Iso 1')
        iso1_button.set_on_clicked(self._on_view_iso1)
        view_orientation_vgrid.add_child(iso1_button)
        
        # Iso 2 view
        iso2_button = gui.Button('Iso 2')
        iso2_button.set_on_clicked(self._on_view_iso2)
        view_orientation_vgrid.add_child(iso2_button)
        
        # X- view
        x_minus_button = gui.Button('X-')
        x_minus_button.set_on_clicked(self._on_view_x_minus)
        view_orientation_vgrid.add_child(x_minus_button)
        
        # Y- view
        y_minus_button = gui.Button('Y-')
        y_minus_button.set_on_clicked(self._on_view_y_minus)
        view_orientation_vgrid.add_child(y_minus_button)

        # Z- view
        z_minus_button = gui.Button('Z-')
        z_minus_button.set_on_clicked(self._on_view_z_minus)
        view_orientation_vgrid.add_child(z_minus_button)
        
        # Iso 3 view
        iso3_button = gui.Button('Iso 3')
        iso3_button.set_on_clicked(self._on_view_iso3)
        view_orientation_vgrid.add_child(iso3_button)
        
        # Iso 4 view
        iso4_button = gui.Button('Iso 4')
        iso4_button.set_on_clicked(self._on_view_iso4)
        view_orientation_vgrid.add_child(iso4_button)
        
        #endregion view orientation buttons

        #################### Section view buttons
        #region section view buttons
        section_view_panel = gui.CollapsableVert('Section view', 5, gui.Margins(5))
        self.view.add_child(section_view_panel)
        
        section_view_vgrid = gui.Horiz(5)
        section_view_panel.add_child(section_view_vgrid)
        
        # Position input
        position_row = gui.Horiz()
        section_view_vgrid.add_child(position_row)
        position_label = gui.Label('Pos:')
        self.position_input = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.position_input.set_value(np.mean(self.build_volume[2])/2)
        self.position_input.set_limits(0.0, self.build_volume[2])
        position_row.add_child(position_label)
        position_row.add_stretch()
        position_row.add_child(self.position_input)
        
        
        
        # X axis section checkbox
        self.x_axis_section_button = gui.Button('x')
        self.x_axis_section_button.set_on_clicked(self._on_section_x)
        section_view_vgrid.add_child(self.x_axis_section_button)
        
        # Y axis section button
        self.y_axis_section_button = gui.Button('y')
        self.y_axis_section_button.set_on_clicked(self._on_section_y)
        section_view_vgrid.add_child(self.y_axis_section_button)
        
        # Z axis section button
        self.z_axis_section_button = gui.Button('z')
        self.z_axis_section_button.set_on_clicked(self._on_section_z)
        section_view_vgrid.add_child(self.z_axis_section_button)
        
        # Off section button
        self.section_off_button = gui.Button('Off')
        self.section_off_button.set_on_clicked(self._off_section)
        self.section_off_button.enabled = False
        section_view_vgrid.add_child(self.section_off_button)
        
        # Flip section button
        self.flip_section_button = gui.Button('Flip')
        self.flip_section_button.set_on_clicked(self._flip_section)
        self.flip_section_button.enabled = False
        section_view_vgrid.add_child(self.flip_section_button)
        
        
        section_view_vgrid.add_stretch()
        
        #endregion section view buttons
           
        # ################### Visibility checkboxes
        #region visibility checkboxes
        self.visibility = gui.CollapsableVert('View control', 5, gui.Margins(5))
        self.view.add_child(self.visibility)
        self.build_volume_view_cb = gui.Checkbox('Build Volume')
        self.build_volume_view_cb.set_on_checked(self._on_toggle_build_volume)
        self.build_volume_view_cb.enabled = True
        self.build_volume_view_cb.checked = True
        self.visibility.add_child(self.build_volume_view_cb)

        
        # Wireframe view
        self.wireframe_view_cb = gui.Checkbox('Wireframe')
        self.wireframe_view_cb.set_on_checked(self._on_toggle_wireframe)
        self.wireframe_view_cb.enabled = False
        self.visibility.add_child(self.wireframe_view_cb)
        

        # Mesh view
        self.mesh_view_cb = gui.Checkbox('Mesh')
        self.mesh_view_cb.set_on_checked(self._on_toggle_mesh_view)
        self.mesh_view_cb.enabled = False
        self.visibility.add_child(self.mesh_view_cb)

        # Voxel view
        self.voxel_view_cb = gui.Checkbox('Voxel')
        self.voxel_view_cb.set_on_checked(self._on_toggle_voxel_view)
        self.voxel_view_cb.enabled = False
        self.visibility.add_child(self.voxel_view_cb)

        # Overhang view
        self.overhang_view_cb = gui.Checkbox('Overhang')
        self.overhang_view_cb.set_on_checked(self._on_toggle_overhang_view)
        self.overhang_view_cb.enabled = False
        self.visibility.add_child(self.overhang_view_cb)

        # Planar view
        self.planar_view = gui.Checkbox('Planar')
        self.planar_view.set_on_checked(self._on_toggle_planar_view)
        self.planar_view.enabled = False
        self.visibility.add_child(self.planar_view)

        # Scalar field view
        self.nonplanar_view_cb = gui.Checkbox('Non-planar')
        self.nonplanar_view_cb.set_on_checked(self._on_toggle_nonplanar_view)
        self.nonplanar_view_cb.enabled = False
        self.visibility.add_child(self.nonplanar_view_cb)

        # Layer view
        self.layer_view_cb = gui.Checkbox('Layer')
        self.layer_view_cb.set_on_checked(self._on_toggle_layer_view)
        self.layer_view_cb.enabled = False
        self.visibility.add_child(self.layer_view_cb)

        # Toolpath view
        self.toolpath_view_cb = gui.Checkbox('Toolpath')
        self.toolpath_view_cb.set_on_checked(self._on_toggle_toolpath_view)
        self.toolpath_view_cb.enabled = False
        self.visibility.add_child(self.toolpath_view_cb)
                
        # Show travel moves checkbox
        self.show_travel_moves = gui.Checkbox('Show Travel Moves')
        self.show_travel_moves.set_on_checked(self._on_toggle_travel_moves)
        self.show_travel_moves.checked = False
        self.visibility.add_child(self.show_travel_moves)

        
        
        #endregion view checkboxes
        
        #################### Sliders (layer, opacity, wireframe linewidth, toolpath linewidth)
        #region sliders
        # Layer slider (both for isosurfaces and toolpaths)
        self.layer_slider = gui.Slider(gui.Slider.INT)
        self.layer_slider.int_value = 0
        self.layer_slider.set_limits(0, 0)
        # self.layer_slider.set_on_value_changed(self._on_isosurface_changed)
        self.layer_slider.set_on_value_changed(self._on_layer_changed)
        self.layer_slider.enabled = False
        self.visibility.add_child(self.layer_slider)

        # Opacity slider    
        opacity_slider = gui.Horiz()
        self.visibility.add_child(opacity_slider)
        opacity_label = gui.Label('Opacity:')
        opacity_slider.add_child(opacity_label)
        opacity_slider.add_stretch()
        self.opacity_setting = gui.Slider(gui.Slider.DOUBLE)
        self.opacity_setting.double_value = 1.0
        self.opacity_setting.set_limits(0.0, 1.0)
        self.opacity_setting.set_on_value_changed(self._on_opacity_changed)
        opacity_slider.add_child(self.opacity_setting)
        opacity_slider.add_fixed(20)

        # Wireframe line width slider
        wireframe_linewidth_slider = gui.Horiz()
        self.visibility.add_child(wireframe_linewidth_slider)
        wireframe_linewidth_label = gui.Label('Wireframe linewidth:')
        wireframe_linewidth_slider.add_child(wireframe_linewidth_label)
        wireframe_linewidth_slider.add_stretch()
        self.wireframe_linewidth_setting = gui.Slider(gui.Slider.DOUBLE)
        self.wireframe_linewidth_setting.double_value = 1.0
        self.wireframe_linewidth_setting.set_limits(0.0, 5.0)
        self.wireframe_linewidth_setting.set_on_value_changed(self._on_wireframe_linewidth_changed)
        wireframe_linewidth_slider.add_child(self.wireframe_linewidth_setting)
        wireframe_linewidth_slider.add_fixed(20)


        # Toolpath line width slider
        toolpath_linewidth_slider = gui.Horiz()
        self.visibility.add_child(toolpath_linewidth_slider)
        toolpath_linewidth_label = gui.Label('Toolpath linewidth:')
        toolpath_linewidth_slider.add_child(toolpath_linewidth_label)
        toolpath_linewidth_slider.add_stretch()
        self.toolpath_linewidth_setting = gui.Slider(gui.Slider.DOUBLE)
        self.toolpath_linewidth_setting.double_value = 1.0
        self.toolpath_linewidth_setting.set_limits(0.0, 100.0)
        self.toolpath_linewidth_setting.set_on_value_changed(self._on_toolpath_line_width_changed)
        toolpath_linewidth_slider.add_child(self.toolpath_linewidth_setting)
        toolpath_linewidth_slider.add_fixed(20)
        #endregion sliders
        
        #################### Load and transform
        #region Load mesh and transform settings
        self.load_mesh_row = gui.Horiz(10)
        self.load.add_child(self.load_mesh_row)
        
        # Center on load checkbox
        self.center_on_load = gui.Checkbox('Center on Load')
        
        self.center_on_load.checked = True
        self.load_mesh_row.add_child(self.center_on_load)
    
    
        # Button to load a file    
        load__mesh_button = gui.Button('Load Mesh')
        load__mesh_button.set_on_clicked(self._on_load_mesh)
        self.load_mesh_row.add_child(load__mesh_button)
        self.load_mesh_row.add_fixed(5)
        
        
        ########### Transform settings
        # Translation
        translation_row = gui.CollapsableVert('Translation', 5, gui.Margins(5))
        self.transform.add_child(translation_row)
        translation_row.set_is_open(False)
        
        self.translation_setting = gui.VectorEdit()
        translation_row.add_child(self.translation_setting)
        self.translation_setting.vector_value = [0, 0, 0]
        self.translation_setting.set_on_value_changed(self._on_translation_changed)
        
        translation_row.add_fixed(5)
        
        # Rotation
        rotation_row = gui.CollapsableVert('Rotation', 5, gui.Margins(5))
        self.transform.add_child(rotation_row)
        rotation_row.set_is_open(False)
        self.rotation_setting = gui.VectorEdit()
        self.rotation_setting.vector_value = [0, 0, 0]
        rotation_row.add_child(self.rotation_setting)
        self.rotation_setting.set_on_value_changed(self._on_rotation_changed)
        rotation_row.add_fixed(5)
        
        # Scale
        scale_row = gui.CollapsableVert('Scale', 5, gui.Margins(5))
        self.transform.add_child(scale_row)
        scale_row.set_is_open(False)
        self.scale_setting = gui.VectorEdit()
        self.scale_setting.vector_value = [1, 1, 1]
        scale_row.add_child(self.scale_setting)
        scale_row.add_fixed(5)
        
        # Center on buildplate button
        center_on_buildplate_button = gui.Button('Center on buildplate')
        center_on_buildplate_button.set_on_clicked(self._on_center_on_buildplate)
        self.transform.add_child(center_on_buildplate_button)
        self.transform.add_fixed(5)
        
        #endregion Load mesh and transform settings
        

        #################### Layer generation parameter inputs
        #region Layer generation parameters
        self.layer_generation_parameters = gui.CollapsableVert('Layer generation parameters', 5, gui.Margins(5))
        self.layer_generation_parameters.set_is_open(False)
        self.layer_generation.add_child(self.layer_generation_parameters)

        # Voxel pitch setting
        pitch_row = gui.Horiz()
        self.layer_generation_parameters.add_child(pitch_row)
        pitch_label = gui.Label('Pitch:')
        pitch_row.add_child(pitch_label)
        pitch_row.add_stretch()
        self.pitch_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.pitch_setting.set_on_value_changed(self._on_pitch_changed)
        self.pitch_setting.set_value(1.0)
        self.pitch_setting.set_limits(0.01, 9999)
        self.pitch_setting.decimal_precision = 2
        pitch_row.add_child(self.pitch_setting)
        pitch_row.add_fixed(20)

        # Angle setting
        angle_row = gui.Horiz()
        self.layer_generation_parameters.add_child(angle_row)
        angle_label = gui.Label('Angle:')
        angle_row.add_child(angle_label)
        angle_row.add_stretch()
        self.angle_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.angle_setting.set_on_value_changed(self._on_angle_changed)
        self.angle_setting.set_value(20.0)
        self.angle_setting.set_limits(0.0, 90.0)
        self.angle_setting.decimal_precision = 1
        angle_row.add_child(self.angle_setting)
        angle_row.add_fixed(20)


        # Overhang thickness
        overhang_thickness_row = gui.Horiz()
        self.layer_generation_parameters.add_child(overhang_thickness_row)
        overhang_thickness_label = gui.Label('Overhang thickness:')
        overhang_thickness_row.add_child(overhang_thickness_label)
        overhang_thickness_row.add_stretch()
        self.overhang_thickness_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.overhang_thickness_setting.set_on_value_changed(self._on_overhang_thickness_changed)
        self.overhang_thickness_setting.set_value(5.0)
        self.overhang_thickness_setting.set_limits(0.0, 9999)
        self.overhang_thickness_setting.decimal_precision = 2
        overhang_thickness_row.add_child(self.overhang_thickness_setting)
        overhang_thickness_row.add_fixed(20)

        # Layer thickness
        layer_thickness_row = gui.Horiz()
        self.layer_generation_parameters.add_child(layer_thickness_row)
        layer_thickness_label = gui.Label('Layer thickness:')
        layer_thickness_row.add_child(layer_thickness_label)
        layer_thickness_row.add_stretch()
        self.layer_thickness_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.layer_thickness_setting.set_on_value_changed(self._on_layer_thickness_changed)
        self.layer_thickness_setting.set_value(0.2)
        self.layer_thickness_setting.set_limits(0.01, 9999)
        self.layer_thickness_setting.decimal_precision = 2
        layer_thickness_row.add_child(self.layer_thickness_setting)
        layer_thickness_row.add_fixed(20)

        # Voxel padding
        voxel_pad_row = gui.Horiz()
        self.layer_generation_parameters.add_child(voxel_pad_row)
        voxel_pad_label = gui.Label('Voxel padding:')
        voxel_pad_row.add_child(voxel_pad_label)
        voxel_pad_row.add_stretch()
        self.voxel_pad_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.INT)
        self.voxel_pad_setting.set_on_value_changed(self._on_voxel_pad_changed)
        self.voxel_pad_setting.set_value(self.overhang_pad)
        self.voxel_pad_setting.set_limits(0, 100)
        voxel_pad_row.add_child(self.voxel_pad_setting)
        voxel_pad_row.add_fixed(20)

        # Overhang pad
        overhang_pad_row = gui.Horiz()
        self.layer_generation_parameters.add_child(overhang_pad_row)
        overhang_pad_label = gui.Label('Overhang pad:')
        overhang_pad_row.add_child(overhang_pad_label)
        overhang_pad_row.add_stretch()
        self.overhang_pad_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.INT)
        self.overhang_pad_setting.set_on_value_changed(self._on_overhang_pad_changed)
        self.overhang_pad_setting.set_value(self.overhang_pad)
        self.overhang_pad_setting.set_limits(0, 100)
        overhang_pad_row.add_child(self.overhang_pad_setting)
        overhang_pad_row.add_fixed(20)
        
        # Smoothing settings
        smoothing_row = gui.Vert()
        self.layer_generation_parameters.add_child(smoothing_row)
        smoothing_label = gui.Label('Smoothing:')
        smoothing_row.add_child(smoothing_label)

        neighbor_label = gui.Label('Neighbors:')
        smoothing_row.add_child(neighbor_label)

        # self.n_neighbors = gui.NumberEdit(o3d.visualization.gui.NumberEdit.INT)
        self.n_neighbors_slider = gui.Slider(gui.Slider.INT)
        self.n_neighbors_slider.int_value = self.smoothing_neighbors
        self.n_neighbors_slider.set_limits(0, 10)
        smoothing_row.add_child(self.n_neighbors_slider)

        smoothing_row.add_fixed(20)

        coeff_label = gui.Label('Coeff:')
        smoothing_row.add_child(coeff_label)

        # self.smoothing_coeff = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.smoothing_coeff_slider = gui.Slider(gui.Slider.DOUBLE)
        self.smoothing_coeff_slider.double_value = 1.0
        self.smoothing_coeff_slider.set_limits(0.0, 1.0)
        smoothing_row.add_child(self.smoothing_coeff_slider)
        smoothing_row.add_fixed(20)


        #endregion Layer generation parameters


        #################### Generate and load isosurfaces
        #region isosurface buttons
        
        # Generate layers button
        self.generate_layers_button = gui.Button('Generate Layers')
        self.generate_layers_button.set_on_clicked(self._on_generate_layers)
        self.generate_layers_button.enabled = False
        self.layer_generation.add_child(self.generate_layers_button)

        # Load isosurfaces button
        self.load_isosurfaces_button = gui.Button('Load Isosurfaces')
        self.load_isosurfaces_button.set_on_clicked(self._on_load_isosurfaces)
        self.load_isosurfaces_button.enabled = False
        self.layer_generation.add_child(self.load_isosurfaces_button)
        #endregion isosurface buttons


        #################### Toolpath  parameters
        #region Toolpath parameter inputs
        self.toolpath_parameters = gui.CollapsableVert('Toolpath Parameters', 5, gui.Margins(5))
        self.toolpath_parameters.set_is_open(False)
        self.toolpath_generation.add_child(self.toolpath_parameters)



        # Toolpath parameter inputs  #
        # Nozzle diameter
        nozzle_diameter_row = gui.Horiz()
        self.toolpath_parameters.add_child(nozzle_diameter_row)
        nozzle_diameter_label = gui.Label('Nozzle Diameter')
        nozzle_diameter_row.add_child(nozzle_diameter_label)
        nozzle_diameter_row.add_stretch()
        self.nozzle_diameter_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.nozzle_diameter_setting.set_value(0.6)
        self.nozzle_diameter_setting.set_limits(0.1, 2.0)
        self.nozzle_diameter_setting.decimal_precision = 2
        nozzle_diameter_row.add_child(self.nozzle_diameter_setting)

        # Number of perimeters
        n_perimeters_row = gui.Horiz()
        self.toolpath_parameters.add_child(n_perimeters_row)
        n_perimeters_label = gui.Label('Perimeters')
        n_perimeters_row.add_child(n_perimeters_label)
        n_perimeters_row.add_stretch()
        self.n_perimeters_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.INT)
        self.n_perimeters_setting.set_value(3)
        n_perimeters_row.add_child(self.n_perimeters_setting)

        # Alternate direction checkbox
        alternate_direction_row = gui.Horiz()
        self.toolpath_parameters.add_child(alternate_direction_row)
        self.alternate_direction_setting = gui.Checkbox('Alternate Direction (cw/ccw)')
        self.alternate_direction_setting.checked = True
        alternate_direction_row.add_child(self.alternate_direction_setting)


        # Seam position
        seam_position_row = gui.Horiz()
        self.toolpath_parameters.add_child(seam_position_row)
        seam_position_label = gui.Label('Seam Position [x, y]')
        seam_position_row.add_child(seam_position_label)
        seam_position_row.add_stretch()
        self.seam_position_setting_x = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.seam_position_setting_x.set_value(0.0)
        self.seam_position_setting_x.decimal_precision = 2
        self.seam_position_setting_x.set_preferred_width(50)
        seam_position_row.add_child(self.seam_position_setting_x)
        seam_position_row.add_fixed(5)
        self.seam_position_setting_y = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.seam_position_setting_y.set_value(0.0)
        self.seam_position_setting_y.decimal_precision = 2
        self.seam_position_setting_y.set_preferred_width(50)
        seam_position_row.add_child(self.seam_position_setting_y)
        
        
        
        # Infill percentage
        infill_percentage_row = gui.Horiz()
        self.toolpath_parameters.add_child(infill_percentage_row)
        infill_percentage_label = gui.Label('Infill %')
        infill_percentage_row.add_child(infill_percentage_label)
        infill_percentage_row.add_stretch()
        self.infill_percentage_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.infill_percentage_setting.set_value(20.0)
        self.infill_percentage_setting.set_limits(0.0, 100.0)
        infill_percentage_row.add_child(self.infill_percentage_setting)

        # Infill perimeter overlap percentage
        infill_perimeter_overlap_row = gui.Horiz()
        self.toolpath_parameters.add_child(infill_perimeter_overlap_row)
        infill_perimeter_overlap_label = gui.Label('Infill Perimeter Overlap %')
        infill_perimeter_overlap_row.add_child(infill_perimeter_overlap_label)
        infill_perimeter_overlap_row.add_stretch()
        self.infill_perimeter_overlap_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.infill_perimeter_overlap_setting.set_value(20.0)
        self.infill_perimeter_overlap_setting.set_limits(0.0, 100.0)
        infill_perimeter_overlap_row.add_child(self.infill_perimeter_overlap_setting)


        # Perimeter line width
        perimeter_line_width_row = gui.Horiz()
        self.toolpath_parameters.add_child(perimeter_line_width_row)
        perimeter_line_width_label = gui.Label('Perimeter Line Width')
        perimeter_line_width_row.add_child(perimeter_line_width_label)
        perimeter_line_width_row.add_stretch()
        self.perimeter_line_width_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.perimeter_line_width_setting.set_value(0.6)
        self.perimeter_line_width_setting.set_limits(0.1, 2.0)
        self.perimeter_line_width_setting.decimal_precision = 2
        perimeter_line_width_row.add_child(self.perimeter_line_width_setting)

        # Infill line width
        infill_line_width_row = gui.Horiz()
        self.toolpath_parameters.add_child(infill_line_width_row)
        infill_line_width_label = gui.Label('Infill Line Width')
        infill_line_width_row.add_child(infill_line_width_label)
        infill_line_width_row.add_stretch()
        self.infill_line_width_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.infill_line_width_setting.set_value(0.6)
        self.infill_line_width_setting.set_limits(0.1, 2.0)
        self.infill_line_width_setting.decimal_precision = 2
        infill_line_width_row.add_child(self.infill_line_width_setting)


        # Filament diameter
        filament_diameter_row = gui.Horiz()
        self.toolpath_parameters.add_child(filament_diameter_row)
        filament_diameter_label = gui.Label('Filament Diameter')
        filament_diameter_row.add_child(filament_diameter_label)
        filament_diameter_row.add_stretch()
        self.filament_diameter_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.filament_diameter_setting.set_value(1.75)
        self.filament_diameter_setting.set_limits(1.0, 3.0)
        self.filament_diameter_setting.decimal_precision = 2
        filament_diameter_row.add_child(self.filament_diameter_setting)

        # Layer fan start   
        layer_fan_start_row = gui.Horiz()
        self.toolpath_parameters.add_child(layer_fan_start_row)
        layer_fan_start_label = gui.Label('Layer Fan Start')
        layer_fan_start_row.add_child(layer_fan_start_label)
        layer_fan_start_row.add_stretch()
        self.layer_fan_start_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.INT)
        self.layer_fan_start_setting.set_value(1)
        self.layer_fan_start_setting.set_limits(0, 9999)
        layer_fan_start_row.add_child(self.layer_fan_start_setting)

        # Z-hop 
        z_hop_row = gui.Horiz()
        self.toolpath_parameters.add_child(z_hop_row)
        z_hop_label = gui.Label('Z-hop')
        z_hop_row.add_child(z_hop_label)
        z_hop_row.add_stretch()
        self.z_hop_setting = gui.NumberEdit(o3d.visualization.gui.NumberEdit.DOUBLE)
        self.z_hop_setting.set_value(0.2)
        self.z_hop_setting.set_limits(0.0, 9999)
        self.z_hop_setting.decimal_precision = 2
        z_hop_row.add_child(self.z_hop_setting)

        #endregion Parameter inputs
        

        #################### Toolpath  buttons
        #region Toolpath generation
        # Generate toolpath button
        self.generate_toolpath_button = gui.Button('Generate Toolpaths')
        self.generate_toolpath_button.set_on_clicked(self._on_generate_toolpaths)
        self.generate_toolpath_button.enabled = False
        self.toolpath_generation.add_child(self.generate_toolpath_button)

        # Load toolpath button
        load_toolpath_button = gui.Button('Load Toolpaths')
        load_toolpath_button.set_on_clicked(self._on_load_toolpaths)
        self.toolpath_generation.add_child(load_toolpath_button)

        #endregion Toolpath generation
        

        # Set the layout callback
        self.window.set_on_layout(self._on_layout)

    def _on_key(self, event):
        if event.key == gui.KeyName.Q:
            self.window.close()
        
        if event.key == gui.KeyName.W and event.type == gui.KeyEvent.DOWN:
            self._on_toggle_wireframe()
            
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene_widget.frame = r
        width = 22 * layout_context.theme.font_size
        height = min(
            r.height,
            self.settings_vert.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self.settings_vert.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        
    def _on_build_volume_changed(self, value):
        self.build_volume = value
        self._on_toggle_build_volume(False)
        self._on_toggle_build_volume(True)
        
    
    # Section view callbacks
    def _on_section_x(self):
        self.x_axis_section_button.enabled = False
        self.y_axis_section_button.enabled = True
        self.z_axis_section_button.enabled = True
        self.section_off_button.enabled = True
        self.flip_section_button.enabled = True
        
        
        self.add_section_view('x', self.position_input.int_value)
            
    def _on_section_y(self):
        self.x_axis_section_button.enabled = True
        self.y_axis_section_button.enabled = False
        self.z_axis_section_button.enabled = True
        self.section_off_button.enabled = True
        self.flip_section_button.enabled = True
    
    def _on_section_z(self):
        self.x_axis_section_button.enabled = True
        self.y_axis_section_button.enabled = True
        self.z_axis_section_button.enabled = False
        self.section_off_button.enabled = True
        self.flip_section_button.enabled = True
        
    def _off_section(self):
        self.x_axis_section_button.enabled = True
        self.y_axis_section_button.enabled = True
        self.z_axis_section_button.enabled = True
        self.section_off_button.enabled = False
        self.flip_section_button.enabled = False
    
    def _flip_section(self):
        pass
    
    # Translation, rotation and scale callbacks
    def _on_center_on_buildplate(self):
        if self.input_mesh is not None:    
            self.center_on_buildplate()
            self.update()
    
    def _on_translation_changed(self, value):
        if self.input_mesh is not None:
            self.T[:3, 3] = value
            self.update()
    
    def _on_rotation_changed(self, value):
        if self.input_mesh is not None:
            ax, ay, az = np.radians(value)
            R = o3d.geometry.get_rotation_matrix_from_xyz([ax, ay, az])
            self.T[:3, :3] = R
            self.update()
            
    def _on_scale_changed(self, value):
        if self.input_mesh is not None:
            self.T[:3, :3] = np.diag(value)
            self.update()
            
    # Loading dialogs and callabacks
    def _on_load_mesh(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, 'Select a file', self.window.theme)
        dlg.add_filter('.stl', 'STL files (.stl)')  # Filter for STL files
        dlg.set_on_done(self._on_load_mesh_done)
        dlg.set_on_cancel(self.window.close_dialog)
        self.window.show_dialog(dlg)

    def _on_load_mesh_done(self, path):
        self.window.close_dialog()
        if not path:
            return

        if self.input_path is not None:
            self.reset()

        try:
            self.input_path = path
            self.input_mesh_filename = os.path.basename(path).split('.')[0]
            self.data_directory = f'{root_data_directory}\\{self.input_mesh_filename}'
            self.output_directory = f'{self.data_directory}\\output'
            self.load_mesh(path)
            self.update()


            self.mesh_view_cb.enabled = True
            self.mesh_view_cb.checked = True
            self.wireframe_view_cb.enabled = True
            
            # self.scene_widget.scene.clear_geometry()
            self._on_toggle_mesh_view(True)

            self.voxel_view_cb.enabled = True
            self.overhang_view_cb.enabled = True
            self.planar_view.enabled = True
            self.nonplanar_view_cb.enabled = True
            self.generate_layers_button.enabled = True
            self.load_isosurfaces_button.enabled = True

        except Exception as e:
            print('Failed to initialize NPLayerGenerator:', e)
    
    def _on_load_isosurfaces(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, 'Select a directory', self.window.theme)
        dlg.set_path(self.output_directory)
        dlg.set_on_done(self._on_load_isosurfaces_done)
        dlg.set_on_cancel(self.window.close_dialog)
        self.window.show_dialog(dlg)

    def _on_load_isosurfaces_done(self, dir):
        self.window.close_dialog()
        if not dir:
            return
            

        try:
            self.isosurface_directory = dir
            isosurfaces = self.layer_generator.load_isosurfaces(dir)
            self.isosurfaces = []
            for surf in isosurfaces:
                surf_o3d = surf.as_open3d
                surf_o3d.compute_triangle_normals()
                surf_o3d.compute_vertex_normals()
                self.isosurfaces.append(surf_o3d)
            # self.add_isosurfaces()
            self.layer_view_cb.enabled = True
            self.layer_view_cb.checked = True
            self._on_toggle_layer_view(True)
            
            self.generate_toolpath_button.enabled = True
            
            # Hide mesh
            self.mesh_view_cb.checked = False
            self._on_toggle_mesh_view(False)
            
            self.scene_widget.force_redraw()
            

            
        except Exception as e:
            print(e)

    def _on_load_toolpaths(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, 'Select a gcode file', self.window.theme)
        if self.output_directory is not None:
            dlg.set_path(self.output_directory)
        else:
            dlg.set_path(root_data_directory)
        dlg.add_filter('.gcode', 'Gcode files (.gcode)')  # Filter for Gcode files
        dlg.set_on_done(self._on_toolpath_dialog_done)
        dlg.set_on_cancel(self.window.close_dialog)
        self.window.show_dialog(dlg)

    def _on_toolpath_dialog_done(self, path):
        self.window.close_dialog()
        if not path:
            return

        try:
            self.add_toolpath(path)
            self.toolpath_view_cb.enabled = True
            self.toolpath_view_cb.checked = True
        except Exception as e:
            print(e)

    # Parameter input callbacks
    def _on_pitch_changed(self, value):
        if self.input_mesh is not None:
            self.pitch = value
            self.update()

    def _on_angle_changed(self, value):
        if self.input_mesh is not None:
            self.angle = value
            self.update()
    
    def _on_overhang_thickness_changed(self, value):
        if self.input_mesh is not None:
            self.overhang_thickness = value
            self.update()
    
    def _on_layer_thickness_changed(self, value):
        if self.input_mesh is not None:
            self.layer_thickness = value
            self.update()
    
    def _on_voxel_pad_changed(self, value):
        if self.input_mesh is not None:
            self.voxel_pad = value
            self.update()
    
    def _on_overhang_pad_changed(self, value):
        if self.input_mesh is not None:
            self.overhang_pad = value
            self.update()
    
    def _on_opacity_changed(self, value):
        try:
            self.material.base_color = np.concatenate((self.material.base_color[:3], [value]))

            for geometry, visible in self.geometry_visibility.items():
                if not visible:
                    continue

                if isinstance(self.geometry_dict[geometry], o3d.geometry.TriangleMesh):
                    self.scene_widget.scene.remove_geometry(geometry)
                    self.scene_widget.scene.add_geometry(geometry, self.geometry_dict[geometry], self.material)
            
            self.scene_widget.force_redraw()
                
        except Exception as e:
            print(e)

    def _on_wireframe_linewidth_changed(self, value):

        try:
            self.line_material.line_width = value
            for geometry, visible in self.geometry_visibility.items():
                if 'wireframe' in geometry and geometry in self.geometry_dict.keys():
                    self.scene_widget.scene.remove_geometry(geometry)
                    self.scene_widget.scene.add_geometry(geometry, self.geometry_dict[geometry], self.line_material)
                    self.scene_widget.scene.show_geometry(geometry, visible)

            self.scene_widget.force_redraw()
        except Exception as e:
            print(e)

    def _on_toolpath_line_width_changed(self, value):
        try:
            self.line_material.line_width = value
            for geometry, visible in self.geometry_visibility.items():
                if not visible:
                    continue
                if 'toolpath' in geometry:
                    self.scene_widget.scene.remove_geometry(geometry)
                    self.scene_widget.scene.add_geometry(geometry, self.geometry_dict[geometry], self.line_material)
        except Exception as e:
            print(e)

# View orientation button callbacks
    def _on_view_x_plus(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([1, 0, 0.2]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
        
    def _on_view_x_minus(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([-1, 0, 0.2]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
        
    def _on_view_y_plus(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([0, 1, 0.2]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
        
    def _on_view_y_minus(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([0, -1, 0.2]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
        
    def _on_view_z_plus(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([0, 0, 1]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 1, 0])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()

    def _on_view_z_minus(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([0, 0, -1]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 1, 0])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
             
    def _on_view_iso1(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([-1, -1, 0.5]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
             
    def _on_view_iso2(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([1, -1, 0.5]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
    
    def _on_view_iso3(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([1, 1, 0.5]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
    
    def _on_view_iso4(self):
        center = self.buildplate_bbox.get_center()
        eye = np.array(center + np.array([-1, 1, 0.5]) * np.linalg.norm(self.buildplate_bbox.get_max_bound() - self.buildplate_bbox.get_center()))
        up = np.array([0, 0, 1])
        
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()
          
    def _on_toggle_wireframe(self):
        self.show_wireframe = not self.show_wireframe   # Flip visibility
        self.wireframe_view_cb.checked = self.show_wireframe
        
        for geometry in self.geometry_visibility.keys():  
            if 'wireframe' in geometry:
                split = geometry.split('_')
                parent_geometry = '_'.join(split[:-1])
                if self.geometry_visibility[parent_geometry]:

                    self.scene_widget.scene.show_geometry(geometry, self.show_wireframe)
                    self.geometry_visibility[geometry] = self.show_wireframe
        self.scene_widget.force_redraw()
        
    def _on_toggle_build_volume(self, is_checked):
        if is_checked:
            # Add build_volume
            self.add_buildvolume()
            
        else:
            self.scene_widget.scene.remove_geometry('build_volume')
            self.scene_widget.scene.remove_geometry('build_plate')
            self.scene_widget.scene.remove_geometry('axes')
            self.geometry_visibility['build_volume'] = False
            self.geometry_visibility['build_plate'] = False
            self.geometry_visibility['axes'] = False
            # Force redraw
        self.scene_widget.force_redraw()
            
#  View checkbox callbacks
    def _on_toggle_mesh_view(self, is_checked):
        if self.input_mesh is None:
            return
            
        try:
            if is_checked:
                self.add_mesh(self.input_mesh, 'mesh')
                self.geometry_visibility['mesh'] = True
                self.geometry_visibility['mesh_wireframe'] = self.show_wireframe
            else:
                self.scene_widget.scene.remove_geometry('mesh')
                self.scene_widget.scene.remove_geometry('mesh_wireframe')
                self.geometry_visibility['mesh'] = False
                self.geometry_visibility['mesh_wireframe'] = False
                self.scene_widget.force_redraw()
                
        except Exception as e:
            print(e)

    def _on_toggle_voxel_view(self, is_checked):
        pass
        try:
            if is_checked:
                self.add_voxels(self.layer_generator.dense, 'voxel_mesh')
            else:
                self.scene_widget.scene.remove_geometry('voxel_mesh')
                self.scene_widget.scene.remove_geometry('voxel_mesh_wireframe')
                
                self.geometry_visibility['voxel_mesh'] = False
                self.geometry_visibility['voxel_mesh_wireframe'] = False
            self.scene_widget.force_redraw()
                
        except Exception as e:
            print(e)

    def _on_toggle_overhang_view(self, is_checked):
        try:
            if is_checked:
                dense = self.layer_generator.dense
                overhang_voxelgrid = self.layer_generator.overhang_mask
                colors = np.ones((overhang_voxelgrid.shape[0], overhang_voxelgrid.shape[1], overhang_voxelgrid.shape[2], 3)) * -1
                
                colors[dense] = self.material.base_color[:3]
                colors[overhang_voxelgrid] = [1, 0, 0]
                colors[colors == -1] = np.nan
                
                self.add_voxels(dense, 'overhang_voxelgrid', colors=colors)
            else:
                for geometry in self.geometry_visibility.keys():
                    if 'overhang_vox' in geometry:
                        self.scene_widget.scene.remove_geometry(geometry)
                        self.geometry_visibility[geometry] = False
                self.scene_widget.force_redraw()
                        
        except Exception as e:
            print(e)

    def _on_toggle_planar_view(self, is_checked):
        if is_checked:
            try:
                dense = self.layer_generator.dense
                planar_scalar_field = self.layer_generator.planar_scalar_field * dense
                
                # Calculate number of layers in the scalar field given the layer thickness 
                s_max = np.max(planar_scalar_field)
                if self.layer_thickness is not None:
                    n = int(np.ceil(s_max))
                else:
                    n = dense.shape[2]
                colormap = NPLayerGenerator.generate_random_cmap(n)
                
                colors = np.nan * np.ones((planar_scalar_field.shape[0], planar_scalar_field.shape[1], planar_scalar_field.shape[2], 3))    
                colors[dense] = colormap(planar_scalar_field[dense]/s_max)[:, :3]  # Map scalar field values to colors
                
                # colors = cm.jet(scalar_field[vis_dense]/np.max(scalar_field[vis_dense]))[:, :3]  # Map scalar field values to colors
                # colors = cm.tab20(scalar_field[vis_dense]/np.max(scalar_field[vis_dense]))[:, :3]  # Map scalar field values to colors
                # colors = cm.viridis(scalar_field[vis_dense]/np.max(scalar_field[vis_dense]))[:, :3]  # Map scalar field values to colors

                self.add_voxels(dense, 'planar_voxelgrid', colors=colors)
                
            except Exception as e:
                print(e)
        else:
            for geometry in self.geometry_visibility.keys():
                if 'planar_vox' in geometry:
                    self.scene_widget.scene.remove_geometry(geometry)
                    self.geometry_visibility[geometry] = False
            self.scene_widget.force_redraw()
                    
    def _on_toggle_nonplanar_view(self, is_checked):
        if is_checked:
            dense = self.layer_generator.dense
            nonplanar_scalar_field = self.layer_generator.scalar_field
            if nonplanar_scalar_field is not None:
                # Calculate number of layers in the scalar field given the layer thickness 
                s_max = np.max(nonplanar_scalar_field)
                n = int(np.ceil(s_max * (self.layer_thickness / self.pitch)))
                colormap = NPLayerGenerator.generate_random_cmap(n+1)

                colors = np.nan * np.ones((nonplanar_scalar_field.shape[0], nonplanar_scalar_field.shape[1], nonplanar_scalar_field.shape[2], 3))
                colors[dense] = colormap(nonplanar_scalar_field[dense]/s_max)[:, :3]  # Map scalar field values to colors

                self.add_voxels(dense, 'nonplanar_voxelgrid', colors=colors)
                

        else:
            for geometry in self.geometry_visibility.keys():
                if 'nonplanar_vox' in geometry:
                    self.scene_widget.scene.remove_geometry(geometry)
                    self.geometry_visibility[geometry] = False
            self.scene_widget.force_redraw()

    def _on_toggle_layer_view(self, is_checked):
        if is_checked:
            self.add_isosurfaces()
            self.layer_slider.enabled = True
            self.layer_slider.set_limits(0, len(self.isosurfaces)-1)
            self.layer_slider.int_value = len(self.isosurfaces)-1
            
        else:
            for i in range(len(self.isosurfaces)):
                self.scene_widget.scene.remove_geometry(f'isosurface_{i}')
                self.geometry_visibility[f'isosurface_{i}'] = False
                if f'isosurface_{i}_wireframe' in self.geometry_dict.keys():
                    self.scene_widget.scene.remove_geometry(f'isosurface_{i}_wireframe')
                    self.geometry_visibility[f'isosurface_{i}_wireframe'] = False
            
            self.layer_slider.enabled = False

            self.scene_widget.force_redraw()
    
    def _on_toggle_toolpath_view(self, is_checked):
        if is_checked:
            self.add_toolpath(self.gcode_path)
            self._on_toolpath_layer_changed(self.layer_slider.int_value)
            
        else:
            for geometry in self.geometry_visibility.keys():
                if 'perimeter' in geometry or 'infill' in geometry or 'travel' in geometry:
                    self.scene_widget.scene.remove_geometry(geometry)
                    self.geometry_visibility[geometry] = False
    
    def _on_layer_changed(self, value):
        self._on_isosurface_changed(value)
        self._on_toolpath_layer_changed(value)
    
    def _on_isosurface_changed(self, value):
        if self.isosurfaces is None:
            return
        
        for i in range(len(self.isosurfaces)):
            is_visible = i <= value

            self.geometry_visibility[f'isosurface_{i}'] = is_visible
            self.scene_widget.scene.show_geometry(f'isosurface_{i}', is_visible)
            if f'isosurface_{i}_wireframe' in self.geometry_dict.keys():
                self.geometry_visibility[f'isosurface_{i}_wireframe'] = is_visible and self.show_wireframe
                self.scene_widget.scene.show_geometry(f'isosurface_{i}_wireframe', is_visible and self.show_wireframe)
        self.scene_widget.force_redraw()
        
    def _on_toggle_travel_moves(self, is_checked):
        if is_checked:
            self._on_toolpath_layer_changed(self.layer_slider.int_value)
            self.scene_widget.force_redraw()
              
    def _on_generate_layers(self):
        try:
            
            if self.isosurfaces is not None:
                for i in range(len(self.isosurfaces)):
                    self.scene_widget.scene.remove_geometry(f'isosurface_{i}')
                    self.scene_widget.scene.remove_geometry(f'isosurface_{i}_wireframe')
                    self.geometry_visibility[f'isosurface_{i}'] = False
                    self.geometry_visibility[f'isosurface_{i}_wireframe'] = False
                    self.isosurfaces = None
                self.scene_widget.force_redraw()
            
            self.update()
            
            self.results_queue = queue.Queue()
            def layer_generation_worker():
                # Disable the generate layers button and change text to 'Generating...'
                

                scalar_field = self.layer_generator.scalar_field
                isovalues = np.arange(1, np.max(scalar_field)+1, 1)
                isosurfaces = self.layer_generator.get_isosurfaces_mp(scalar_field, isovalues, n=self.n_neighbors, c=self.smoothing_coeff, export=False)
                isosurfaces = self.layer_generator.trim(isosurfaces, export=True)
                self.results_queue.put(isosurfaces)
                isosurfaces = self.results_queue.get()
                self.isosurfaces = []
                for surf in isosurfaces:
                    surf_o3d = surf.as_open3d
                    surf_o3d.compute_triangle_normals()
                    surf_o3d.compute_vertex_normals()
                    self.isosurfaces.append(surf_o3d)

                # Enable layer view and visualize the isosurfaces
                self.layer_view_cb.enabled = True
                self.layer_view_cb.checked = True
                self._on_toggle_layer_view(True)
                
                # Re-enable the generate layers button and change text back to 'Generate Layers'
                self.generate_layers_button.enabled = True
                self.generate_layers_button.text = 'Generate Layers'
                
                self.generate_toolpath_button.enabled = True
                self.isosurface_directory = self.output_directory + f'\\p{self.pitch}\\lt{self.layer_thickness}mm'
                self.number_of_layers = len(self.isosurfaces)
                # Hide mesh
                self.mesh_view_cb.checked = False
                self._on_toggle_mesh_view(False)

            self.thread = threading.Thread(target=layer_generation_worker)
            self.thread.start()
        except Exception as e:
            print(e)
            

    def _on_generate_toolpaths(self):
        self.result_queue = queue.Queue()
        filename = self.input_mesh_filename
        self.gcode_path = self.data_directory + f'\\output\\{filename}_p{self.pitch}_lt{self.layer_thickness}_{datetime.now().strftime("%d.%m.%y_%H.%M")}.gcode'  # Output path for the toolpath, will be saved as a .gcode file
        translate_xy = self.T[:2, 3]
    
        def toolpath_generation_worker():
            
            generate_toolpaths( self.isosurface_directory,
                                self.gcode_path,
                                self.layer_thickness_setting.double_value,
                                self.infill_percentage_setting.double_value,
                                self.nozzle_diameter_setting.double_value,
                                self.perimeter_line_width_setting.double_value,
                                self.infill_line_width_setting.double_value,
                                self.n_perimeters_setting.int_value,
                                self.alternate_direction_setting.checked,
                                self.infill_perimeter_overlap_setting.double_value,
                                self.z_hop_setting.double_value,
                                self.filament_diameter_setting.double_value,
                                self.layer_fan_start_setting.int_value,
                                [self.seam_position_setting_x.double_value, self.seam_position_setting_y.double_value],
                                translate_xy=translate_xy
            )

            self.add_toolpath(self.gcode_path)
            self._on_toggle_toolpath_view(True)
        
            self.toolpath_view_cb.enabled = True
            self.toolpath_view_cb.checked = True
            self.layer_slider.enabled = True
            self.scene_widget.force_redraw()
        
        self.thread = threading.Thread(target=toolpath_generation_worker)
        self.thread.start()

    def _on_toolpath_layer_changed(self, value):
        try:
            if self.number_of_layers is None:
                return
            
            for i in range(self.number_of_layers):
                self.scene_widget.scene.remove_geometry(f'perimeter_layer_{i}')
                self.scene_widget.scene.remove_geometry(f'infill_layer_{i}')
                self.scene_widget.scene.remove_geometry(f'travel_layer_{i}')
                self.geometry_visibility[f'perimeter_layer_{i}'] = False
                self.geometry_visibility[f'infill_layer_{i}'] = False
                self.geometry_visibility[f'travel_layer_{i}'] = False

                if i <= value:
                    if i < value:
                        self.geometry_dict[f'perimeter_layer_{i}'].paint_uniform_color(np.array(self.perimeter_color) * 0.25)
                        self.geometry_dict[f'infill_layer_{i}'].paint_uniform_color(np.array(self.infill_color) * 0.25)
                        self.geometry_dict[f'travel_layer_{i}'].paint_uniform_color(np.array(self.travel_color) * 0.25)
                    
                    self.scene_widget.scene.add_geometry(f'perimeter_layer_{i}', self.geometry_dict[f'perimeter_layer_{i}'], self.line_material)
                    self.scene_widget.scene.add_geometry(f'infill_layer_{i}', self.geometry_dict[f'infill_layer_{i}'], self.line_material)
                    self.scene_widget.scene.add_geometry(f'travel_layer_{i}', self.geometry_dict[f'travel_layer_{i}'], self.line_material)
                    
                    self.scene_widget.scene.show_geometry(f'perimeter_layer_{i}', True)
                    self.scene_widget.scene.show_geometry(f'infill_layer_{i}', True)
                    self.scene_widget.scene.show_geometry(f'travel_layer_{i}', self.show_travel_moves.checked)
                    
                    self.geometry_visibility[f'perimeter_layer_{i}'] = True
                    self.geometry_visibility[f'infill_layer_{i}'] = True
                    self.geometry_visibility[f'travel_layer_{i}'] = self.show_travel_moves.checked
            self.scene_widget.force_redraw()
        
        except Exception as e:
            print(e)
            
 #### ADD GEOMETRY TO SCENE ####
    def add_buildvolume(self):
        try:
            # Build volume line material
            build_volume_line_material = o3d.visualization.rendering.MaterialRecord()
            build_volume_line_material.shader = 'unlitLine'
            build_volume_line_material.line_width = 1.5
            build_volume_line_material.base_color = [.25, .25, .25, 1]
            
            buildvolume = self.build_volume
            buildvolume_vertices = np.array([
                                            [0, 0, 0],
                                            [buildvolume[0], 0, 0],
                                            [buildvolume[0], buildvolume[1], 0],
                                            [0, buildvolume[1], 0],
                                            [0, 0, buildvolume[2]],
                                            [buildvolume[0], 0, buildvolume[2]],
                                            [buildvolume[0], buildvolume[1], buildvolume[2]],
                                            [0, buildvolume[1], buildvolume[2]]
                                            ]).astype(np.float64)
            # build_volume_vertices -= np.array([build_volume[0]/2, build_volume[1]/2, 0])
            
            lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

            buildvolume_lineset = o3d.geometry.LineSet()
            buildvolume_lineset.points = o3d.utility.Vector3dVector(buildvolume_vertices)
            buildvolume_lineset.lines = o3d.utility.Vector2iVector(lines)  
            # buildvolume_lineset.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]] * len(lines)))
            # buildvolume_lineset.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]] * len(lines)))
            
            self.scene_widget.scene.add_geometry('build_volume', buildvolume_lineset, build_volume_line_material)
            self.scene_widget.scene.show_geometry('build_volume', True)
            self.geometry_dict['build_volume'] = buildvolume_lineset
            self.geometry_visibility['build_volume'] = True

            
            
            
            # Build plate lineset
            build_plate_vertices = []
            build_plate_lines = []
            
            cell_size = 10
            nx = int(buildvolume[0]/cell_size)+1
            ny = int(buildvolume[1]/cell_size)+1
            
            for i in range(0, nx):
                p1 = [i*cell_size, 0, 0]    
                p2 = [i*cell_size, buildvolume[1], 0]
                build_plate_vertices.extend([p1, p2])
                build_plate_lines.append([len(build_plate_vertices)-2, len(build_plate_vertices)-1])
                
            for i in range(0, ny):
                p1 = [0, i*cell_size, 0]
                p2 = [buildvolume[0], i*cell_size, 0]
                
                build_plate_vertices.extend([p1, p2])
                build_plate_lines.append([len(build_plate_vertices)-2, len(build_plate_vertices)-1])
                
            build_plate_vertices = np.array(build_plate_vertices).astype(np.float64)
            # build_plate_vertices -= np.array([build_volume[0]/2, build_volume[1]/2, 0])
            # build_plate_vertices -= np.array(build_volume) * n_lines//cell_size/4 # Center the build plate
            
            build_plate_vertices[:, 2] = 0
            
            build_plate_lineset = o3d.geometry.LineSet()
            build_plate_lineset.points = o3d.utility.Vector3dVector(build_plate_vertices)
            build_plate_lineset.lines = o3d.utility.Vector2iVector(build_plate_lines)
            # build_plate_lineset.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]] * len(build_plate_lines)))
            
            self.scene_widget.scene.add_geometry('build_plate', build_plate_lineset, build_volume_line_material)
            self.scene_widget.scene.show_geometry('build_plate', True)
            self.geometry_dict['build_plate'] = build_plate_lineset
            self.geometry_visibility['build_plate'] = True
            
            # Add coordinate axes
            # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[-build_volume[0]/2, -build_volume[1]/2, 0])
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
            self.scene_widget.scene.add_geometry('axes', axes, self.material)
            self.geometry_dict['axes'] = axes
            self.geometry_visibility['axes'] = True

            self.scene_widget.force_redraw()
        except Exception as e:
            print(e)
            
    def add_voxels(self, dense, name, color=None, colors=None):
        # colors: 3D array of RGB values for each voxel
        voxel_size = self.pitch*0.8
        
        
        points = self.tensor_to_point_cloud(dense)
        points = points - self.layer_generator.transform
        
        if colors is not None:
            unique_colors = np.unique(colors[~np.isnan(colors)].reshape(-1, 3), axis=0)
            for color in unique_colors:
                colored_mask = np.all(colors == color, axis=-1) 
                self.add_voxels(colored_mask, f'{name}_{color}', color=color)   # Recursively add voxels for each color
                
        
        else:
            voxel_mesh = o3d.geometry.TriangleMesh()
            for p in points:
                voxel = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
                voxel.translate(p)
                voxel_mesh += voxel
            
            if color is None:
                color = [0.8, 0.8, 0.8]
                
            self.add_mesh(voxel_mesh, name, color)
                
    def add_mesh(self, mesh, mesh_name, color=[.2, .3, 1], apply_transform=True):
        mesh_copy = copy.deepcopy(mesh)
        
        mesh_copy.compute_triangle_normals()
        mesh_copy.compute_vertex_normals()
        
        if apply_transform:
            mesh_copy.transform(self.T)

        mesh_copy.paint_uniform_color(np.array(color))
        mesh_copy.compute_vertex_normals()
        mesh_copy.compute_triangle_normals()
        
        self.scene_widget.scene.add_geometry(mesh_name, mesh_copy, self.material)
        self.geometry_dict[mesh_name] = mesh_copy
        self.geometry_visibility[mesh_name] = True

        # Add wireframe
        self.add_wireframe(mesh_name)
        
        self.scene_widget.force_redraw()

    def add_wireframe(self, geometry, color=[0.2, 0.2, 0.2]):
        wireframe_material = o3d.visualization.rendering.MaterialRecord()
        wireframe_material.shader = 'defaultUnlit'
        wireframe_material.line_width = self.wireframe_linewidth_setting.double_value
        
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry_dict[geometry])    
        
        colors = np.array([color for i in range(len(wireframe.lines))])
        wireframe.colors = o3d.utility.Vector3dVector(colors)
        
        self.scene_widget.scene.add_geometry(f'{geometry}_wireframe', wireframe, self.line_material)
        self.scene_widget.scene.show_geometry(f'{geometry}_wireframe', self.show_wireframe)
        self.geometry_dict[f'{geometry}_wireframe'] = wireframe
        self.geometry_visibility[f'{geometry}_wireframe'] = self.show_wireframe
        
        self.scene_widget.force_redraw()

    def add_isosurfaces(self):
        if self.isosurfaces is None:
            return
        
        for i, isosurface in enumerate(self.isosurfaces):
            if isosurface is not None:
                color = np.random.rand(3)
                self.add_mesh(isosurface, f'isosurface_{i}', color)
                
    def add_toolpath(self, path, show_travel=False):
        self.perimeter_color = [0, 0, 1]
        self.infill_color = [0, 1, 0]
        self.travel_color = [1, 0, 0]
        
        perimeter_line_segments = {-1:[]} # -1 is the movement before the first layer is printed
        perimeter_extrusion_lengths = {-1:[]}
        
        infill_line_segments = {-1:[]} 
        infill_extrusion_lengths = {-1:[]}
        
        travel_line_segments = {-1:[]} 
        
        with open(path, 'r') as f:
            current_position = [0, 0, 0]
            current_layer = -1
            for line in f:
                # 
                if 'Retraction' in line:
                    continue

                if line.startswith(';Start layer'):
                    current_layer = int(line.split(' ')[2])
                    perimeter_line_segments[current_layer] = []
                    perimeter_extrusion_lengths[current_layer] = []
                    infill_line_segments[current_layer] = []
                    infill_extrusion_lengths[current_layer] = []
                    
                    travel_line_segments[current_layer] = []
                    continue

                if line.startswith('G1'):
                    split = re.split(' ', line)
                    for s in split:
                        if s.startswith('X'):
                            x = float(s[1:])
                        elif s.startswith('Y'):
                            y = float(s[1:])
                        elif s.startswith('Z'):
                            z = float(s[1:])
                        elif s.startswith('E'):
                            e = float(s[1:])
                    if 'X' not in line:
                        x = current_position[0]
                    if 'Y' not in line:
                        y = current_position[1]
                    if 'Z' not in line:
                        z = current_position[2]
                    if 'E' not in line:
                        e = 0

                    if 'Perimeter' in line:
                        perimeter_line_segments[current_layer].append([current_position, [x, y, z]])
                        perimeter_extrusion_lengths[current_layer].append(e if e else 0)
                        current_position = [x, y, z]    
                        continue
                    if 'Infill' in line:
                        infill_line_segments[current_layer].append([current_position, [x, y, z]])
                        infill_extrusion_lengths[current_layer].append(e if e else 0)
                        current_position = [x, y, z]    
                        continue
                    if 'Travel' in line or e <= 0:
                        travel_line_segments[current_layer].append([current_position, [x, y, z]])
                        current_position = [x, y, z]    
                        continue
                        
                    
        
        for layer_number in perimeter_line_segments.keys():
            perimeter_layer_lines = perimeter_line_segments[layer_number]
            infill_layer_lines = infill_line_segments[layer_number]
            travel_layer_lines = travel_line_segments[layer_number]
            
            perimeter_points = []
            perimeter_lines = []
            for line in perimeter_layer_lines:
                perimeter_points.extend(line)
                perimeter_lines.append([len(perimeter_points)-2, len(perimeter_points)-1])
            perimeter_colors = [self.perimeter_color for i in range(len(perimeter_lines)) ]
            
            infill_points = []
            infill_lines = []
            for line in infill_layer_lines:
                infill_points.extend(line)
                infill_lines.append([len(infill_points)-2, len(infill_points)-1])
            infill_colors = [self.infill_color for i in range(len(infill_lines)) ]
            
            travel_points = []
            travel_lines = []
            for line in travel_layer_lines:
                travel_points.extend(line)
                travel_lines.append([len(travel_points)-2, len(travel_points)-1])
            travel_colors = [self.travel_color for i in range(len(travel_lines)) ]
            
            
            
            
            perimeter_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(perimeter_points),
                lines=o3d.utility.Vector2iVector(perimeter_lines),
            )
            perimeter_line_set.colors = o3d.utility.Vector3dVector(perimeter_colors)
            self.scene_widget.scene.add_geometry(f'perimeter_layer_{layer_number}', perimeter_line_set, self.line_material)
            
            infill_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(infill_points),
                lines=o3d.utility.Vector2iVector(infill_lines),
            )
            infill_line_set.colors = o3d.utility.Vector3dVector(infill_colors)
            self.scene_widget.scene.add_geometry(f'infill_layer_{layer_number}', infill_line_set, self.line_material)
            
            
            travel_line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(travel_points),
                lines=o3d.utility.Vector2iVector(travel_lines),
            )
            travel_line_set.colors = o3d.utility.Vector3dVector(travel_colors)
            self.scene_widget.scene.add_geometry(f'travel_layer_{layer_number}', travel_line_set, self.line_material)
            
            self.geometry_dict[f'perimeter_layer_{layer_number}'] = perimeter_line_set
            self.geometry_dict[f'infill_layer_{layer_number}'] = infill_line_set
            self.geometry_dict[f'travel_layer_{layer_number}'] = travel_line_set
            
            self.geometry_visibility[f'perimeter_layer_{layer_number}'] = True
            self.geometry_visibility[f'infill_layer_{layer_number}'] = True
            self.geometry_visibility[f'travel_layer_{layer_number}'] = show_travel
            
            
            if not show_travel:
                self.scene_widget.scene.remove_geometry(f'travel_layer_{layer_number}')
                
                
            
        self.number_of_layers = np.max(list(perimeter_line_segments.keys())) + 1
        
        print('Loaded', self.number_of_layers, 'layers')
        # self.num_layers_label.text = f'Loaded {self.number_of_layers} layers'
    
        self.layer_slider.set_limits(-1, self.number_of_layers-1)
        self.layer_slider.int_value = self.number_of_layers-1
        self.layer_slider.enabled = True

        self._on_toolpath_layer_changed(self.number_of_layers-1)
        self.scene_widget.force_redraw()
 
    def center_on_buildplate(self):
        if self.input_mesh is not None:
            bbox = self.input_mesh.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            translation = np.array([self.build_volume[0]/2, self.build_volume[1]/2, 0]) + np.array([center[0], center[1], -bbox.min_bound[2]])
            self.T[:3, 3] = translation
            self.translation_setting.vector_value = translation
 
    def load_mesh(self, path):
        
        if self.input_mesh is not None:   # Delete the previous mesh
            self.scene_widget.scene.remove_geometry('mesh')
            self.scene_widget.scene.remove_geometry('mesh_wireframe')
            self.geometry_visibility['mesh'] = False
            self.geometry_visibility['mesh_wireframe'] = False
            
        try:
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh:
                # Center mesh on orgin and place on z=0 plane
                bbox = mesh.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                mesh.translate(np.array([-center[0], -center[1], -bbox.min_bound[2]]))

                self.input_mesh = mesh        
                
                if self.center_on_load.checked:
                    self.center_on_buildplate()
                self.add_mesh(self.input_mesh, 'mesh')
                
                         
        except Exception as e:
            print(e)

    def tensor_to_point_cloud(self, tensor):
        points = []
        
        indices = np.argwhere(tensor)
        points = indices * self.layer_generator.pitch

        return np.array(points) # Scale point by pitch to get real world coordinates

    def initialize_layer_generator(self):
        self.layer_generator = NPLayerGenerator.NPLayerGenerator(self.input_path,
                                                                 root_data_directory, 
                                                                 self.pitch, 
                                                                 self.layer_thickness)
        
        self.layer_generator.generate_scalar_field(self.angle, 
                                                   self.overhang_thickness, 
                                                   self.voxel_pad, 
                                                   self.overhang_pad)

    def update(self):
        self.pitch = self.pitch_setting.double_value
        self.layer_thickness = self.layer_thickness_setting.double_value
        self.angle = self.angle_setting.double_value
        self.overhang_thickness = self.overhang_thickness_setting.double_value
        self.voxel_pad = self.voxel_pad_setting.int_value
        self.overhang_pad = self.overhang_pad_setting.int_value
        self.n_neighbors = self.n_neighbors_slider.int_value
        self.smoothing_coeff = self.smoothing_coeff_slider.int_value
        

        self.initialize_layer_generator()

        self.scene_widget.scene.clear_geometry()
        self._on_toggle_mesh_view(self.mesh_view_cb.checked)
        self._on_toggle_voxel_view(self.voxel_view_cb.checked)
        self._on_toggle_overhang_view(self.overhang_view_cb.checked)
        self._on_toggle_planar_view(self.planar_view.checked)
        self._on_toggle_nonplanar_view(self.nonplanar_view_cb.checked)
        self._on_toggle_layer_view(self.layer_view_cb.checked)
        self._on_toggle_toolpath_view(self.toolpath_view_cb.checked)
        self._on_toggle_build_volume(self.build_volume_view_cb.checked)
        
    def reset(self):
        self.scene_widget.scene.clear_geometry()
        # self.scene_widget.scene.remove_geometry('mesh')
        # self.scene_widget.scene.remove_geometry('voxel_grid')
        # self.scene_widget.scene.remove_geometry('filled_pcd')
        # self.scene_widget.scene.remove_geometry('highlight_pcd')
        # self.scene_widget.scene.remove_geometry('planar_voxelgrid')
        # self.scene_widget.scene.remove_geometry('nonplanar_voxelgrid')
        # for i in range(len(self.isosurfaces)):
        #     self.scene_widget.scene.remove_geometry(f'isosurface_{i}')
        
        
        self.input_mesh = None
        self.mesh_view_cb.checked = True
        self.voxel_view_cb.checked = False
        self.overhang_view_cb.checked = False
        self.planar_view.checked = False
        self.nonplanar_view_cb.checked = False
        self.layer_view_cb.checked = False

    def add_section_view(self, axis=None, pos=0):
        # axis: 'x', 'y', 'z'
        # pos: position of intersection plane along the axis
        # returns a list of geometries intersected by the plane
        
        
        
        scene_bbox = self.scene_widget.scene.bounding_box
        
        if axis == 'x':
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound = [0, 0, 0], 
                                                       max_bound = [pos, self.build_volume[1], self.build_volume[2]])
        elif axis == 'y':
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound = [0, 0, 0], 
                                                       max_bound = [self.build_volume[0], pos, self.build_volume[2]])
        elif axis == 'z':
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound = [0, 0, 0], 
                                                       max_bound = [self.build_volume[0], self.build_volume[1], pos])
        else:
            return
        
        
        
        cropped_geometries = {}
        for name, geometry in self.geometry_dict.items():
            if not isinstance(geometry, o3d.geometry.TriangleMesh):
                continue
            if name == 'axes':
                continue
            
            if 'crop' in name:
                self.scene_widget.scene.remove_geometry(name)
                self.geometry_visibility[name] = False
            cropped_geometries[name] = o3d.geometry.TriangleMesh.crop(geometry, bbox)
            
            
        for name, geometry in cropped_geometries.items():
            self.geometry_dict[name+'_cropped'] = geometry
            self.geometry_visibility[name+'_cropped'] = self.geometry_visibility[name]
            
        

        
def main():
    try:
        gui.Application.instance.initialize()
        NPSlicer()
        gui.Application.instance.run()
    except Exception as e:
        print(e)
        
if __name__ == '__main__':
    main()
