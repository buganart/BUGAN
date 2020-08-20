#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :]
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path of or folder with.obj files")
parser.add_argument("--output", help="Rendered image output path prefix")
parser.add_argument("--energy", help="Brightness", default=2, type=float)
parser.add_argument("--border", help="Zoom", default=0.1, type=float)
parser.add_argument("--color", help="Color", default="custom", type=str)
parser.add_argument("--samples", help="Samples", default=None, type=int)
args = parser.parse_args(argv)

camera_y = -8
camera_z = 0.907267
energy = args.energy

scene = bpy.context.scene

scene.render.tile_x = 64
scene.render.tile_y = 64

bpy.context.scene.render.image_settings.quality = 90
bpy.context.scene.render.image_settings.file_format = "JPEG"
bpy.context.scene.render.resolution_x = 500
bpy.context.scene.render.resolution_y = 500

#### Set up cycle rendering with CUDA #####
scene.cycles.device = "GPU"
prefs = bpy.context.preferences
prefs.addons["cycles"].preferences.get_devices()
cprefs = prefs.addons["cycles"].preferences

# Attempt to set GPU device types if available
for compute_device_type in ("CUDA", "OPENCL", "NONE"):
    try:
        cprefs.compute_device_type = compute_device_type
        print("SET", compute_device_type)
        break
    except TypeError:
        pass

# Enable all CPU and GPU devices
for device in cprefs.devices:
    print("DEVICE TYPE", device, device.type)
    if device.type == "CUDA":
        device.use = True
        scene.render.tile_x = 256
        scene.render.tile_y = 256
        if args.samples:
            scene.cycles.samples = args.samples

#### Clean up #####
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

##### Create and position camera #####
bpy.ops.object.camera_add(
    enter_editmode=False, align="VIEW", location=(0, 0, 0), rotation=(1.5708, 0, 0),
)

cam = bpy.data.objects[bpy.context.active_object.name]
bpy.context.scene.camera = cam

bpy.ops.transform.translate(
    value=(0, camera_y, camera_z),
    orient_type="GLOBAL",
    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    orient_matrix_type="GLOBAL",
    constraint_axis=(False, True, False),
    mirror=True,
    use_proportional_edit=False,
    proportional_edit_falloff="SMOOTH",
    proportional_size=1,
    use_proportional_connected=False,
    use_proportional_projected=False,
)

##### Create and position key light ######
bpy.ops.object.light_add(type="SUN", radius=1, location=(0, 0, 0))
bpy.ops.transform.rotate(
    value=-0.777158,
    orient_axis="Y",
    orient_type="GLOBAL",
    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    orient_matrix_type="GLOBAL",
    constraint_axis=(False, True, False),
    mirror=True,
    use_proportional_edit=False,
    proportional_edit_falloff="SMOOTH",
    proportional_size=1,
    use_proportional_connected=False,
    use_proportional_projected=False,
)
bpy.context.object.data.energy = energy

##### Create and position fill light ######
bpy.ops.object.duplicate_move(
    OBJECT_OT_duplicate={"linked": False, "mode": "TRANSLATION"},
    TRANSFORM_OT_translate={
        "value": (0, 0, 0),
        "orient_type": "GLOBAL",
        "orient_matrix": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "orient_matrix_type": "GLOBAL",
        "constraint_axis": (False, False, False),
        "mirror": True,
        "use_proportional_edit": False,
        "proportional_edit_falloff": "SMOOTH",
        "proportional_size": 1,
        "use_proportional_connected": False,
        "use_proportional_projected": False,
        "snap": False,
        "snap_target": "CLOSEST",
        "snap_point": (0, 0, 0),
        "snap_align": False,
        "snap_normal": (0, 0, 0),
        "gpencil_strokes": False,
        "cursor_transform": False,
        "texture_space": False,
        "remove_on_cancel": False,
        "release_confirm": False,
        "use_accurate": False,
    },
)
bpy.ops.transform.rotate(
    value=3.13598,
    orient_axis="Y",
    orient_type="GLOBAL",
    orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    orient_matrix_type="GLOBAL",
    constraint_axis=(False, True, False),
    mirror=True,
    use_proportional_edit=False,
    proportional_edit_falloff="SMOOTH",
    proportional_size=1,
    use_proportional_connected=False,
    use_proportional_projected=False,
)
bpy.context.object.data.energy = energy / 5

##### Black background #####
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
    0,
    0,
    0,
    1,
)


def move_camera_away(border):
    bpy.ops.object.select_all(action="DESELECT")
    cam = bpy.data.objects["Camera"]
    cam.select_set(True)
    old_y = cam.location[1]
    new_y = old_y * (1 + border)
    delta_y = new_y - old_y
    bpy.ops.transform.translate(
        value=(0, delta_y, 0),
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        constraint_axis=(False, True, False),
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )


def render(mesh_path):
    ##### Load and scale tree #####
    bpy.ops.import_scene.obj(filepath=str(mesh_path.resolve()))
    bpy.ops.transform.resize(
        value=(0.02, 0.02, 0.02),
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )

    for mat in bpy.data.materials:
        print("mat", mat)
        if mat.node_tree and args.color != "default":
            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
                0.260611,
                0.363661,
                0.8,
                1,
            )
    bpy.context.object.active_material_index = 0
    bpy.ops.object.material_slot_assign()

    bpy.context.scene.render.filepath = str(
        Path(args.output).joinpath(f"{Path(mesh_path).stem}")
    )

    # Select objects that will be rendered
    bpy.ops.object.select_all(action="DESELECT")
    # for obj in bpy.context.scene.objects:
    #     obj.select_set(False)
    for obj in bpy.context.visible_objects:
        obj.select_set(True)
    bpy.ops.view3d.camera_to_view_selected()

    move_camera_away(args.border)

    bpy.ops.render.render(write_still=True)

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[-1].select_set(True)
    bpy.ops.object.delete()


input_path_arg = Path(args.input)

input_paths = (
    sorted(Path(args.input).glob("*.obj"))
    if input_path_arg.expanduser().is_dir()
    else [input_path_arg]
)

for input_path in input_paths:
    print(f"{input_path} -> {args.output}")
    try:
        render(input_path)
    except Exception as exc:
        print(exc)
