"""Blender Add-on for the lipsync animation
"""
import csv
from math import pi, sqrt
import os
from pathlib import Path
from typing import Optional, Set, Tuple
import aud
import bpy
from bpy_extras.io_utils import ExportHelper


# Add-on information
bl_info = {
    "name": "Lipsync",
    "author": "Inkyu",
    "version": (0, 5, 2),
    "blender": (3, 4, 0),
    "location": "View3D > Sidebar > Lipsync",
    "description": "Tools for generating lipsync animation",
    "category": "Lipsync",
}

# Global variables
BASIS_KEY = "Basis"
DIFF_MATERIAL = "DiffMaterial"
ACTION_NAME = "LipsyncAction"
DIFF_ACTION_NAME = "DiffAction"


class LipsyncProperty(bpy.types.PropertyGroup):
    """Store the properties"""

    audio_path_mesh_sequence: bpy.props.StringProperty(
        name="Audio",
        default="",
        description="Path of the audio file",
        subtype="FILE_PATH",
    )

    mesh_sequence_dir: bpy.props.StringProperty(
        name="Mesh Sequence",
        default="",
        description="Directory of the mesh sequence",
        subtype="DIR_PATH",
    )

    neutral_path: bpy.props.StringProperty(
        name="Template Mesh",
        default="",
        description="Path of the template mesh",
        subtype="FILE_PATH",
    )

    blendshape_dir: bpy.props.StringProperty(
        name="Blendshape Meshes",
        default="",
        description="Directory of the blendshape meshes",
        subtype="DIR_PATH",
    )

    audio_path_blendshape: bpy.props.StringProperty(
        name="Speech",
        default="",
        description="Path of the audio file",
        subtype="FILE_PATH",
    )

    blendshape_weights_path: bpy.props.StringProperty(
        name="Blendshape Coefficient Sequence",
        default="",
        description="Path of the blendshape coefficient sequence (CSV)",
        subtype="FILE_PATH",
    )

    fps_mesh: bpy.props.IntProperty(
        name="FPS",
        default=60,
        description="Automatically set fps when -1",
    )

    fps_blendshape: bpy.props.IntProperty(
        name="FPS",
        default=60,
        description="Automatically set fps when -1",
    )

    rearrange_blendshape: bpy.props.BoolProperty(
        name="Rearrange Blendshapes",
        default=True,
        description="Rearrange the order of shape keys (Blendshapes)",
    )

    target_obj: bpy.props.PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Target object to visualize the vertex differences",
    )

    color_multiplier: bpy.props.FloatProperty(
        name="Color Multiplier",
        default=1.0,
        description="Multiplier for the color which visualize the vertex differences",
    )

    vis_option: bpy.props.EnumProperty(
        name="Visualize Option",
        items=[
            ("Vector", "Vector", ""),
            ("Amplitude", "Amplitude", ""),
        ],
        default="Vector",
        description="Option for the visualization",
    )


class Lipsync_PT_MeshsequencePanel(bpy.types.Panel):
    """Mesh panel"""

    bl_idname = "LIPSYNC_PT_meshsequence_panel"
    bl_label = "Mesh Sequence Panel"
    bl_category = "Lipsync"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context: bpy.types.Context):
        """Draw UI elements

        Parameters
        ----------
        context : bpy.types.Context
            Blender context
        """
        box_mesh = self.layout.box()

        row = box_mesh.row()
        row.prop(context.scene.lipsync_property, "audio_path_mesh_sequence")

        row = box_mesh.row()
        row.prop(context.scene.lipsync_property, "mesh_sequence_dir")

        row = box_mesh.row()
        row.prop(context.scene.lipsync_property, "fps_blendshape")

        # Button
        row = box_mesh.row()
        row.operator("lipsync.generate_mesh_anime_operator", text="Import Facial Motion")


class Lipsync_PT_BlendshapePanel(bpy.types.Panel):
    """Mesh panel"""

    bl_idname = "LIPSYNC_PT_blendshape_panel"
    bl_label = "Blendshape Panel"
    bl_category = "Lipsync"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context: bpy.types.Context):
        """Draw UI elements

        Parameters
        ----------
        context : bpy.types.Context
            Blender context
        """
        # Box for blendshape
        box_blendshape = self.layout.box()

        row = box_blendshape.row()
        row.prop(context.scene.lipsync_property, "neutral_path")

        row = box_blendshape.row()
        row.prop(context.scene.lipsync_property, "blendshape_dir")

        row = box_blendshape.row()
        row.operator("lipsync.import_blendshape_operator", text="Import Blendshape Facial Model")

        # Box for generating anime
        box_anime = self.layout.box()

        # Visualize the selected object
        row = box_anime.row()
        row.label(text="Selected Object: ", icon="OBJECT_DATA")
        box = box_anime.box()
        selected_objects = context.selected_objects
        if len(selected_objects) == 0:
            box.label(text="", icon="KEYFRAME")
        else:
            box.label(text=selected_objects[0].name, icon="KEYFRAME")

        row = box_anime.row()
        row.prop(context.scene.lipsync_property, "audio_path_blendshape")

        row = box_anime.row()
        row.prop(context.scene.lipsync_property, "blendshape_weights_path")

        row = box_anime.row()
        row.prop(context.scene.lipsync_property, "fps_blendshape")

        row = box_anime.row()
        row.prop(context.scene.lipsync_property, "rearrange_blendshape")

        row = box_anime.row()
        row.operator(
            "lipsync.generate_blendshape_anime_operator",
            text="Import Facial Motion",
        )

        # Box for saving anime
        box_save = self.layout.box()

        row = box_save.row()
        row.label(text="Selected Object: ", icon="OBJECT_DATA")
        box = box_save.box()
        if len(selected_objects) == 0:
            box.label(text="", icon="KEYFRAME")
        else:
            box.label(text=selected_objects[0].name, icon="KEYFRAME")

        row = box_save.row()
        row.operator(
            "lipsync.save_blendshape_anime_operator",
            text="Save Facial Motion",
        )

        box_visualize = self.layout.box()

        row = box_visualize.row()
        row.prop(context.scene.lipsync_property, "target_obj")

        row = box_visualize.row()
        row.prop(context.scene.lipsync_property, "vis_option")

        row = box_visualize.row()
        row.prop(context.scene.lipsync_property, "color_multiplier")

        row = box_visualize.row()
        row.operator(
            "lipsync.visualize_difference_operator",
            text="Visualize Difference",
        )


def load_obj(context: bpy.types.Context, path: str) -> Optional[bpy.types.Object]:
    """Load object from the path. Currently support 'ply' and 'obj' file.

    Parameters
    ----------
    context : bpy.types.Context
        Blender context

    path: str
        Path of the object file

    Returns
    -------
    Optional[bpy.types.Object]
        Loaded object
    """
    ext = os.path.splitext(path)[-1]

    obj = None

    if ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
        obj = context.object
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=path, split_mode="OFF")
        obj = context.selected_objects[0]

    return obj


def load_speaker(
    context: bpy.types.Context, audio_path: str
) -> Tuple[bpy.types.Object, int]:
    """Load audio using speaker object

    Parameters
    ----------
    context : bpy.types.Context
        Blender context

    audio_path: str
        Path of the audio file

    Returns
    -------
    Tuple[bpy.types.Object, int]
        (Speaker object, length of the audio)
    """
    # Load the sound
    bpy.ops.object.speaker_add(rotation=(pi, 0.0, 0.0))
    speaker = context.object
    speaker.hide_set(True)
    speaker.data.sound = bpy.data.sounds.load(audio_path)
    speaker.data.attenuation = 0.0
    speaker.data.update_tag()

    sd = aud.Sound(audio_path)
    length = sd.length
    del sd

    return speaker, length


class LipsyncGenerateMeshAnimeOperator(bpy.types.Operator):
    """Operator for the 'Import Facial Motion' button in Meshsequence panel"""

    bl_idname = "lipsync.generate_mesh_anime_operator"
    bl_label = "lipsync.generate_mesh_anime_operator"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute the operator

        Parameters
        ----------
        context : bpy.types.Context
            Blender context

        Returns
        -------
        Set[str]
            Set of status messages: https://docs.blender.org/api/current/bpy.types.Operator.html
        """
        audio_path = bpy.path.abspath(
            context.scene.lipsync_property.audio_path_mesh_sequence
        )
        mesh_sequence_dir = bpy.path.abspath(
            context.scene.lipsync_property.mesh_sequence_dir
        )

        # Reset the scene frame to 1
        context.scene.frame_set(1)

        # Load the sound
        speaker, length = load_speaker(context, audio_path)

        # List the mesh sequence
        sequence = []
        for path in sorted(os.listdir(mesh_sequence_dir)):
            abs_path = os.path.join(mesh_sequence_dir, path)
            if os.path.isfile(abs_path):
                sequence.append(abs_path)

        # Update frame rate of the animation
        num_sequence = len(sequence)

        fps = context.scene.lipsync_property.fps_mesh
        if fps < 0:
            samplerate = speaker.data.sound.samplerate
            context.scene.render.fps = num_sequence
            context.scene.render.fps_base = length / samplerate
        else:
            context.scene.render.fps = fps
            context.scene.render.fps_base = 1

        # Generate animation sequence
        obj = load_obj(context, sequence[0])
        if obj is None:
            self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
            return {"CANCELLED"}

        mesh = obj.data
        num_vertices = len(mesh.vertices)

        # Generate vertices coordinates lists
        coords_list = []

        # Default vertices coordinates
        coords = [None] * (num_vertices * 3)
        mesh.vertices.foreach_get("co", coords)
        coords_list.append(coords)

        # Next vertices coordinates
        for sdx in range(1, num_sequence):
            obj_tmp = load_obj(context, sequence[sdx])
            if obj_tmp is None:
                self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
                return {"CANCELLED"}

            coords = [None] * (num_vertices * 3)
            obj_tmp.data.vertices.foreach_get("co", coords)
            coords_list.append(coords)

            bpy.ops.object.delete()

        # Create animation_data, action
        mesh.animation_data_create()
        mesh.animation_data.action = bpy.data.actions.new(name=ACTION_NAME)

        frames = range(1, num_sequence + 1)

        # Insert keyframes
        for vdx in range(num_vertices):
            for idx in range(3):
                fcurve = mesh.animation_data.action.fcurves.new(
                    data_path=f"vertices[{vdx}].co",
                    index=idx,
                )

                samples = [
                    coords_list[fdx][3 * vdx + idx] for fdx in range(num_sequence)
                ]

                fcurve.keyframe_points.add(count=num_sequence)
                fcurve.keyframe_points.foreach_set(
                    "co", [x for co in zip(frames, samples) for x in co]
                )

        # Update the frame end
        context.scene.frame_end = max(context.scene.frame_end, num_sequence)

        return {"FINISHED"}


class LipsyncAddBlendshapeOperator(bpy.types.Operator):
    """Operator for the 'Add Blendshape' button in Blendshape panel"""

    bl_idname = "lipsync.import_blendshape_operator"
    bl_label = "lipsync.import_blendshape_operator"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute the operator

        Parameters
        ----------
        context : bpy.types.Context
            Blender context

        Returns
        -------
        Set[str]
            Set of status messages: https://docs.blender.org/api/current/bpy.types.Operator.html
        """
        neutral_path = bpy.path.abspath(context.scene.lipsync_property.neutral_path)
        blendshape_dir = bpy.path.abspath(context.scene.lipsync_property.blendshape_dir)

        # Get blendshape names and absolute paths
        blendshape_names = []
        blendshape_paths = []
        for path in sorted(os.listdir(blendshape_dir)):
            abs_path = os.path.join(blendshape_dir, path)
            if os.path.isfile(abs_path):
                blendshape_names.append(Path(path).stem)
                blendshape_paths.append(abs_path)

        # Load neutral mesh
        obj = load_obj(context, neutral_path)
        if obj is None:
            self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
            return {"CANCELLED"}

        num_vertices = len(obj.data.vertices)

        # Create shape keys
        obj.shape_key_add(name=BASIS_KEY)

        for bdx in range(len(blendshape_names)):
            obj_bdx = load_obj(context, blendshape_paths[bdx])
            if obj_bdx is None:
                self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
                continue

            sk = obj.shape_key_add(name=blendshape_names[bdx])

            coords = [None] * (num_vertices * 3)
            obj_bdx.data.vertices.foreach_get("co", coords)
            sk.data.foreach_set("co", coords)

            bpy.ops.object.delete()

        return {"FINISHED"}


class LipsyncGenerateBlendshapeAnimeOperator(bpy.types.Operator):
    """Operator for the 'Generate Blendshape Anime' button in Blendshape panel"""

    bl_idname = "lipsync.generate_blendshape_anime_operator"
    bl_label = "lipsync.generate_blendshape_anime_operator"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute the operator

        Parameters
        ----------
        context : bpy.types.Context
            Blender context

        Returns
        -------
        Set[str]
            Set of status messages: https://docs.blender.org/api/current/bpy.types.Operator.html
        """
        audio_path = bpy.path.abspath(
            context.scene.lipsync_property.audio_path_blendshape
        )
        blendshape_weights_path = bpy.path.abspath(
            context.scene.lipsync_property.blendshape_weights_path
        )

        selected_objects = context.selected_objects
        if len(selected_objects) == 0:
            self.report({"ERROR_INVALID_INPUT"}, "Should have to select the object")
            return {"CANCELLED"}
        obj = selected_objects[0]

        # Reset the scene frame to 1
        context.scene.frame_set(1)

        # Load the sound
        speaker, length = load_speaker(context, audio_path)

        # Reset the active object
        context.view_layer.objects.active = obj

        # Load blendshape weights
        weights = []
        with open(blendshape_weights_path, "r") as csvfile:
            rdr = csv.DictReader(csvfile)
            for row in rdr:
                for k, v in row.items():
                    row[k] = float(v)
                weights.append(row)

        # Update frame rate of the animation
        num_sequence = len(weights)
        samplerate = speaker.data.sound.samplerate

        fps = context.scene.lipsync_property.fps_blendshape
        if fps < 0:
            samplerate = speaker.data.sound.samplerate
            context.scene.render.fps = num_sequence
            context.scene.render.fps_base = length / samplerate
        else:
            context.scene.render.fps = fps
            context.scene.render.fps_base = 1

        key_blocks = obj.data.shape_keys.key_blocks

        # Change the order of shape_keys
        if context.scene.lipsync_property.rearrange_blendshape and num_sequence > 0:
            blendshape_keys = list(weights[0].keys())
            blendshape_keys.insert(0, BASIS_KEY)

            for key in reversed(blendshape_keys):
                kdx = key_blocks.find(key)
                if kdx != -1:
                    obj.active_shape_key_index = kdx
                    bpy.ops.object.shape_key_move(type="TOP")

        # Generate animation sequence
        for wdx, weight in enumerate(weights):
            for keyblock in key_blocks:
                if keyblock.name == BASIS_KEY:
                    continue

                if keyblock.name in weight:
                    keyblock.value = weight[keyblock.name]
                    keyblock.keyframe_insert("value", frame=wdx + 1)

        # Update the frame end
        context.scene.frame_end = max(context.scene.frame_end, num_sequence)

        return {"FINISHED"}


class LipsyncSaveBlendshapeAnimeOperator(bpy.types.Operator, ExportHelper):
    """Operator for the 'Save Blendshape Anime' button in Blendshape panel"""

    bl_idname = "lipsync.save_blendshape_anime_operator"
    bl_label = "Save Blendshape"

    filename_ext = ".csv"
    filter_glob: bpy.props.StringProperty(
        default="*.csv",
        options={"HIDDEN"},
    )

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute the operator

        Parameters
        ----------
        context : bpy.types.Context
            Blender context

        Returns
        -------
        Set[str]
            Set of status messages: https://docs.blender.org/api/current/bpy.types.Operator.html
        """
        selected_objects = context.selected_objects
        if len(selected_objects) == 0:
            self.report({"ERROR_INVALID_INPUT"}, "No selected object")
            return {"CANCELLED"}

        obj = selected_objects[0]
        key_blocks = obj.data.shape_keys.key_blocks

        blendshape_keys = key_blocks.keys()[1:]  # Remove BASIS_KEY

        vals = []

        fcurves = obj.data.shape_keys.animation_data.action.fcurves
        if fcurves:
            num_frame = len(fcurves[0].keyframe_points)
            vals = [[0 for _ in range(len(blendshape_keys))] for _ in range(num_frame)]

        for fcurve in fcurves:
            key = fcurve.data_path.split('"')[1]
            if key == BASIS_KEY:
                continue

            kdx = blendshape_keys.index(key)

            for fdx, point in enumerate(fcurve.keyframe_points):
                frame_val = point.co[1]
                vals[fdx][kdx] = frame_val

        with open(self.filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(blendshape_keys)
            writer.writerows(vals)

        return {"FINISHED"}


class LipsyncVisualizeDifferenceOperator(bpy.types.Operator):
    """Operator for the 'Visualize Difference' button in Blendshape panel"""

    bl_idname = "lipsync.visualize_difference_operator"
    bl_label = "Visualize Difference"

    def execute(self, context: bpy.types.Context) -> Set[str]:
        """Execute the operator

        Parameters
        ----------
        context : bpy.types.Context
            Blender context

        Returns
        -------
        Set[str]
            Set of status messages: https://docs.blender.org/api/current/bpy.types.Operator.html
        """
        selected_objects = context.selected_objects
        if len(selected_objects) == 0:
            self.report({"ERROR_INVALID_INPUT"}, "No selected object")
            return {"CANCELLED"}

        obj = selected_objects[0]

        target_obj = context.scene.lipsync_property.target_obj
        if target_obj is None:
            self.report({"ERROR_INVALID_INPUT"}, "No target object")
            return {"CANCELLED"}

        ### Update vertex colors
        mesh = obj.data

        if len(mesh.vertex_colors) == 0:
            mesh.vertex_colors.new()

        # Set material
        if DIFF_MATERIAL not in bpy.data.materials:
            mat = bpy.data.materials.new(name=DIFF_MATERIAL)
        mat = bpy.data.materials[DIFF_MATERIAL]
        mat.use_nodes = True

        vcolor = mat.node_tree.nodes.new("ShaderNodeVertexColor")

        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(vcolor.outputs[0], bsdf.inputs[0])

        mesh.materials.append(mat)

        new_mat_idx = len(mesh.materials) - 1

        for poly in mesh.polygons:
            poly.material_index = new_mat_idx

        color_multiplier = context.scene.lipsync_property.color_multiplier
        vis_option = context.scene.lipsync_property.vis_option

        try:
            fcurves = obj.data.shape_keys.animation_data.action.fcurves
            num_frame = len(fcurves[0].keyframe_points)
        except:
            num_frame = 0

        # Compute the difference
        diff_dict = [{} for _ in range(num_frame)]
        for fdx in range(num_frame):
            context.scene.frame_set(fdx + 1)
            for poly in mesh.polygons:
                for i, vdx in enumerate(poly.vertices):
                    diff = mesh.vertices[vdx].co - target_obj.data.vertices[vdx].co
                    loop_idx = poly.loop_indices[i]
                    diff_dict[fdx][loop_idx] = diff

        context.scene.frame_set(1)

        # Create animation_data, action
        mesh.animation_data_create()
        mesh.animation_data.action = bpy.data.actions.new(name=DIFF_ACTION_NAME)

        frames = range(1, num_frame + 1)

        # Insert keyframes
        for poly in mesh.polygons:
            for i, vdx in enumerate(poly.vertices):
                loop_idx = poly.loop_indices[i]
                diff = [diff_dict[fdx][loop_idx] for fdx in range(num_frame)]

                diff_val = []
                if vis_option == "Vector":
                    diff_val = [(abs(df[0]), abs(df[1]), abs(df[2])) for df in diff]
                elif vis_option == "Amplitude":
                    amp = [sqrt(df[0] ** 2 + df[1] ** 2 + df[2] ** 2) for df in diff]
                    diff_val = [(ap, ap, ap) for ap in amp]
                else:
                    self.report({"ERROR_INVALID_INPUT"}, "Weird visualize option")
                    return {"CANCELLED"}

                for idx in range(3):
                    fcurve = mesh.animation_data.action.fcurves.new(
                        data_path=f"vertex_colors.active.data[{loop_idx}].color",
                        index=idx,
                    )

                    samples = [
                        diff_val[fdx][idx] * color_multiplier
                        for fdx in range(num_frame)
                    ]

                    fcurve.keyframe_points.add(count=num_frame)
                    fcurve.keyframe_points.foreach_set(
                        "co", [x for co in zip(frames, samples) for x in co]
                    )

        return {"FINISHED"}


# List of classes to be registered
classes = (
    LipsyncProperty,
    Lipsync_PT_MeshsequencePanel,
    Lipsync_PT_BlendshapePanel,
    LipsyncGenerateMeshAnimeOperator,
    LipsyncAddBlendshapeOperator,
    LipsyncGenerateBlendshapeAnimeOperator,
    LipsyncSaveBlendshapeAnimeOperator,
    LipsyncVisualizeDifferenceOperator,
)


def register():
    """Register the classes"""
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lipsync_property = bpy.props.PointerProperty(type=LipsyncProperty)


def unregister():
    """Unregister the classes"""
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lipsync_property


if __name__ == "__main__":
    register()
