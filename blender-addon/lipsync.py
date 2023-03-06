import csv
from math import pi
import os
from pathlib import Path
from typing import Optional, Set, Tuple
import aud
import bpy


bl_info = {
    "name": "Lipsync",
    "author": "Inkyu",
    "version": (0, 2, 1),
    "blender": (3, 4, 0),
    "location": "View3D > Sidebar > Lipsync",
    "description": "Tools for generating lipsync animation",
    "category": "Lipsync",
}


class LipsyncProperty(bpy.types.PropertyGroup):
    """Store the properties"""

    audio_path_mesh_sequence: bpy.props.StringProperty(
        name="Audio Path",
        default="",
        description="Path of the audio file",
        subtype="FILE_PATH",
    )

    mesh_sequence_dir: bpy.props.StringProperty(
        name="Mesh Sequence Dir",
        default="",
        description="Directory of the mesh sequence",
        subtype="DIR_PATH",
    )

    neutral_path: bpy.props.StringProperty(
        name="Neutral Path",
        default="",
        description="Path of the neutral mesh",
        subtype="FILE_PATH",
    )

    blendshape_dir: bpy.props.StringProperty(
        name="Blendshape Path",
        default="",
        description="Directory of the blendshapes",
        subtype="DIR_PATH",
    )

    audio_path_blendshape: bpy.props.StringProperty(
        name="Audio Path",
        default="",
        description="Path of the audio file",
        subtype="FILE_PATH",
    )

    blendshape_weights_path: bpy.props.StringProperty(
        name="Blendshape Weights",
        default="",
        description="Path of the blendshape weights (CSV)",
        subtype="FILE_PATH",
    )


class Lipsync_PT_MeshsequencePanel(bpy.types.Panel):
    """Mesh panel"""

    bl_idname = "LIPSYNC_PT_meshsequence_panel"
    bl_label = "Mesh Sequence Panel"
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
        row = self.layout.row()
        row.prop(context.scene.lipsync_property, "audio_path_mesh_sequence")

        row = self.layout.row()
        row.prop(context.scene.lipsync_property, "mesh_sequence_dir")

        # Button
        row = self.layout.row()
        row.operator("lipsync.generate_mesh_anime_operator", text="Generate Anime")


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
        row.operator("lipsync.import_blendshape_operator", text="Import Blendshape")

        # Box for anime
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
        row.operator(
            "lipsync.generate_blendshape_anime_operator",
            text="Generate Anime",
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
    """Operator for the 'Generate Anime' button in Meshsequence panel"""

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

        # Create collection
        collection = bpy.data.collections.new("Lipsync")
        context.scene.collection.children.link(collection)

        # Load the sound
        speaker, length = load_speaker(context, audio_path)
        context.collection.objects.unlink(speaker)
        collection.objects.link(speaker)

        # List the mesh sequence
        sequence = []
        for path in sorted(os.listdir(mesh_sequence_dir)):
            abs_path = os.path.join(mesh_sequence_dir, path)
            if os.path.isfile(abs_path):
                sequence.append(abs_path)

        # Update frame rate of the animation
        num_sequence = len(sequence)
        samplerate = speaker.data.sound.samplerate

        context.scene.render.fps = num_sequence
        context.scene.render.fps_base = length / samplerate

        # Generate animation sequence
        obj = load_obj(context, sequence[0])
        if obj is None:
            self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
            return {"CANCELLED"}

        context.collection.objects.unlink(obj)
        collection.objects.link(obj)
        obj.name = "Object"
        obj.data.name = "Object"

        num_vertices = len(obj.data.vertices)

        for sdx in range(1, num_sequence):
            obj_tmp = load_obj(context, sequence[sdx])
            if obj_tmp is None:
                self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
                return {"CANCELLED"}

            for vdx in range(num_vertices):
                obj.data.vertices[vdx].co = obj_tmp.data.vertices[vdx].co
                obj.data.vertices[vdx].keyframe_insert("co", frame=sdx + 1)

            bpy.ops.object.delete()

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
        obj.shape_key_add(name="Basis")

        for bdx in range(len(blendshape_names)):
            obj_bdx = load_obj(context, blendshape_paths[bdx])
            if obj_bdx is None:
                self.report({"ERROR_INVALID_INPUT"}, "File format is not supported")
                continue

            sk = obj.shape_key_add(name=blendshape_names[bdx])
            for vdx in range(num_vertices):
                for i in range(3):
                    sk.data[vdx].co[i] = obj_bdx.data.vertices[vdx].co[i]

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

        # Create collection
        collection = bpy.data.collections.new("Lipsync")
        context.scene.collection.children.link(collection)

        selected_objects = context.selected_objects
        if len(selected_objects) == 0:
            self.report({"ERROR_INVALID_INPUT"}, "Should have to select the object")
            return {"CANCELLED"}
        obj = selected_objects[0]

        context.collection.objects.unlink(obj)
        collection.objects.link(obj)

        # Reset the scene frame to 1
        context.scene.frame_set(1)

        # Load the sound
        speaker, length = load_speaker(context, audio_path)
        context.collection.objects.unlink(speaker)
        collection.objects.link(speaker)

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

        context.scene.render.fps = num_sequence
        context.scene.render.fps_base = length / samplerate

        # Generate animation sequence
        for wdx, weight in enumerate(weights):
            for keyblock in obj.data.shape_keys.key_blocks:
                if keyblock.name == "Basis":
                    continue

                if keyblock.name in weight:
                    keyblock.value = weight[keyblock.name]
                    keyblock.keyframe_insert("value", frame=wdx + 1)

        return {"FINISHED"}


# List of classes to be registered
classes = (
    LipsyncProperty,
    Lipsync_PT_MeshsequencePanel,
    Lipsync_PT_BlendshapePanel,
    LipsyncGenerateMeshAnimeOperator,
    LipsyncAddBlendshapeOperator,
    LipsyncGenerateBlendshapeAnimeOperator,
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
