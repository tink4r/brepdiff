"""Render a STEP-converted NPZ file with Blender.

Usage::

    $ blender --background --python scripts/blender/render_step.py -- \
        {npz_path} {render_path} {color}

Where ``npz_path`` is produced by :mod:`scripts.visualization.vis_step`,
``render_path`` is the desired PNG output path and ``color`` is one of
``blue``, ``pink``, ``orange`` or ``green``.

The meshes contained in the NPZ are assumed to be normalised to the
``[-1, 1]`` range and already positioned so that the lowest point sits on
``z=-0.5``.  This matches the processing done in ``vis_step.py`` and in the
viewer utilities.
"""

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List

import bpy
import numpy as np


# Ensure we run on Blender 3.4 – the version used throughout the project.
_BLENDER_VERSION = bpy.app.version
assert (
    _BLENDER_VERSION[0] == 3 and _BLENDER_VERSION[1] == 4
), f"The Blender version is {_BLENDER_VERSION[0]}.{_BLENDER_VERSION[1]}.x, expected 3.4"


@dataclass(frozen=True)
class Arguments:
    npz_path: str
    render_path: str
    color_name: str


def parse_arguments() -> Arguments:
    """Parse CLI arguments passed after ``--`` by Blender."""

    parameters = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(parameters) < 3:
        raise ValueError(
            "Expected arguments: {npz_path} {render_path} {color}; received "
            f"{parameters}"
        )

    return Arguments(
        npz_path=parameters[0],
        render_path=parameters[1],
        color_name=parameters[2].lower(),
    )


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load the NPZ file exported by ``vis_step`` utilities."""

    data = dict(np.load(path, allow_pickle=True))
    return data


def cleanup_scene() -> None:
    """Remove all existing objects from the current scene."""

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def get_color_rgba(name: str) -> np.ndarray:
    """Map a user-provided colour name to an RGBA tuple."""

    color_map = {
        "blue": (0.231, 0.388, 0.933, 1.0),
        "pink": (0.933, 0.372, 0.717, 1.0),
        "orange": (0.984, 0.573, 0.129, 1.0),
        "green": (0.313, 0.725, 0.435, 1.0),
    }

    if name not in color_map:
        allowed = ", ".join(sorted(color_map))
        raise ValueError(f"Unsupported colour '{name}'. Expected one of: {allowed}")

    return color_map[name]


def create_material(name: str, color: Iterable[float]) -> bpy.types.Material:
    """Create a simple Principled BSDF material with the given colour."""

    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = tuple(color)
    bsdf.inputs["Specular"].default_value = 0.2
    bsdf.inputs["Roughness"].default_value = 0.5
    return material


def create_mesh_object(vertices: np.ndarray, triangles: np.ndarray) -> bpy.types.Object:
    """Create and link a mesh object from vertices and triangle indices."""

    mesh = bpy.data.meshes.new("BREPMesh")

    verts_list: List[List[float]] = vertices.tolist() if vertices.size else []
    faces_list: List[List[int]] = triangles.tolist() if triangles.size else []
    mesh.from_pydata(verts_list, [], faces_list)
    mesh.update()

    obj = bpy.data.objects.new("BREPObject", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    if faces_list:
        bpy.ops.object.shade_smooth()

    return obj


def create_edge_objects(edges: Iterable[np.ndarray], material: bpy.types.Material) -> None:
    """Create tube-like curve objects for every polyline edge."""

    for edge_id, edge_points in enumerate(edges):
        if edge_points is None:
            continue

        edge_array = np.asarray(edge_points)
        if edge_array.ndim != 2 or edge_array.shape[0] < 2:
            continue

        curve_data = bpy.data.curves.new(f"EdgeCurve{edge_id}", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.use_fill_caps = True
        curve_data.bevel_depth = 0.01

        spline = curve_data.splines.new(type="POLY")
        spline.points.add(edge_array.shape[0] - 1)
        for idx, coord in enumerate(edge_array):
            spline.points[idx].co = (float(coord[0]), float(coord[1]), float(coord[2]), 1.0)

        curve_object = bpy.data.objects.new(f"EdgeObject{edge_id}", curve_data)
        curve_object.data.materials.append(material)
        bpy.context.collection.objects.link(curve_object)


def create_shadow_catcher(size: float = 6.0) -> bpy.types.Object:
    """Create a large plane that acts as a shadow catcher."""

    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, -1.0))
    plane = bpy.context.active_object
    plane.name = "ShadowCatcherPlane"

    material = bpy.data.materials.new(name="ShadowCatcherMaterial")
    material.use_nodes = True
    plane.data.materials.append(material)
    plane.is_shadow_catcher = True
    return plane


def configure_lighting() -> None:
    """Set up a simple three-point lighting rig."""

    bpy.ops.object.light_add(type="SUN", location=(4.0, -2.0, 4.0))
    key_light = bpy.context.active_object
    key_light.data.energy = 2.0

    bpy.ops.object.light_add(type="AREA", location=(-4.0, 1.5, 3.0))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 20.0
    fill_light.scale = (3.0, 3.0, 3.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, 4.0, 3.5))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 12.0
    rim_light.scale = (2.5, 2.5, 2.5)


def configure_camera() -> bpy.types.Object:
    """Create and orient a camera looking at the scene."""

    bpy.ops.object.camera_add(location=(2.4, -2.8, 2.2))
    camera = bpy.context.active_object
    camera.rotation_euler = (0.95, 0.0, 0.785398)  # ~55° downward, 45° around Z
    bpy.context.scene.camera = camera
    return camera


def configure_render_settings(output_path: str) -> None:
    """Adjust render settings for consistent results."""

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True
    scene.render.filepath = output_path
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.device = "GPU"

    # Try to enable GPU rendering when available.
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    for device in preferences.devices:
        if device.type in {"CUDA", "OPTIX"}:
            device.use = True

    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGB"

    # Composite over a white background for reproducible outputs.
    scene.use_nodes = True
    compositor = scene.node_tree
    compositor.nodes.clear()
    render_layers = compositor.nodes.new(type="CompositorNodeRLayers")
    white_bg = compositor.nodes.new(type="CompositorNodeRGB")
    white_bg.outputs["RGBA"].default_value = (1.0, 1.0, 1.0, 1.0)
    alpha_over = compositor.nodes.new(type="CompositorNodeAlphaOver")
    composite = compositor.nodes.new(type="CompositorNodeComposite")

    compositor.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    compositor.links.new(white_bg.outputs["RGBA"], alpha_over.inputs[1])
    compositor.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])

    scene.view_settings.view_transform = "Standard"


def main() -> None:
    args = parse_arguments()
    color = get_color_rgba(args.color_name)

    cleanup_scene()

    data = load_npz(args.npz_path)
    vertices = data.get("vertices", np.empty((0, 3), dtype=float))
    triangles = data.get("triangles", np.empty((0, 3), dtype=int))
    edges_raw = data.get("edges", [])

    mesh_material = create_material("BREPMaterial", color)
    edge_material = create_material("BREPEdgeMaterial", color)
    edge_material.diffuse_color = color

    mesh_object = create_mesh_object(vertices, triangles)
    if mesh_material not in mesh_object.data.materials:
        mesh_object.data.materials.append(mesh_material)
    mesh_object.active_material = mesh_material

    edges_iterable: Iterable[np.ndarray]
    if isinstance(edges_raw, np.ndarray):
        edges_iterable = edges_raw.tolist()
    else:
        edges_iterable = edges_raw

    create_edge_objects(edges_iterable, edge_material)

    create_shadow_catcher()
    configure_lighting()
    configure_camera()
    configure_render_settings(args.render_path)

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
 
