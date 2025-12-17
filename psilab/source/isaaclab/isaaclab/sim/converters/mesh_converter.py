# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os

import omni
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdUtils, UsdShade, Sdf

from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import export_prim_to_file
# from isaaclab.sim.utils import safe_set_attribute_on_usd_prim


class MeshConverter(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        The USD file has Y-up axis and is scaled to meters.
        The asset hierarchy is arranged as follows:

        .. code-block:: none
            mesh_file_basename (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Use new structure for RL here:
                /file_name (default prim)
                |- /Looks
                    |- /material
                        |- /Shader
                |- /base_link 
                    |- /mesh
                    or 
                    |- /visuals
                    |- /collisions

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()

        # Check if mesh_file_basename is a valid USD identifier
        if not Tf.IsValidIdentifier(mesh_file_basename):
            # Correct the name to a valid identifier and update the basename
            mesh_file_basename_original = mesh_file_basename
            mesh_file_basename = Tf.MakeValidIdentifier(mesh_file_basename)
            omni.log.warn(
                f"Input file name '{mesh_file_basename_original}' is an invalid identifier for the mesh prim path."
                f" Renaming it to '{mesh_file_basename}' for the conversion."
            )

        # Convert USD
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(in_file=cfg.asset_path, out_file=self.usd_path)
        )
        # Create a new stage, set Z up and meters per unit
        temp_stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)
        # Add mesh to stage
        base_prim = temp_stage.DefinePrim(f"/{mesh_file_basename}", "Xform")
        # (Kaisa) Define a new prim and add all references under it begin
        # prim = temp_stage.DefinePrim(f"/{mesh_file_basename}/geometry", "Xform")
        base_link_prim = temp_stage.DefinePrim(f"/{mesh_file_basename}/base_link", "Xform")
        base_link_prim.GetReferences().AddReference(self.usd_path)
        # (Kaisa) Define a new prim and add all references under it end
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(self.usd_path)

        # Open converted USD stage
        stage = Usd.Stage.Open(self.usd_path)
        # Need to reload the stage to get the new prim structure, otherwise it can be taken from the cache
        stage.Reload()
        # Add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"

        # (Kaisa) Move Looks begin
        # (Kaisa) Since "MovePrim" doesn't have the attribute "stage", we need to set the context manually
        context = omni.usd.get_context()
        context.open_stage(self.usd_path) 

        xform_prim = stage.GetDefaultPrim()
        # geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/geometry")
        geom_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/base_link")
        old_looks_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}/base_link/Looks")
        if old_looks_prim.IsValid():
            # (Kaisa) Use "MovePrim" instead of "MovePrimCommand", as the latter seems to have a bug
            result = omni.kit.commands.execute("MovePrim", 
                path_from=old_looks_prim.GetPath(),
                path_to=f"{xform_prim.GetPath().pathString}/Looks",
                # stage_or_context=stage 
            )
            print("Move result:", result)

        # (Kaisa) Rename material and Shader
        looks_path = f"{xform_prim.GetPath().pathString}/Looks"
        self.safe_hierarchical_rename(stage, looks_path)
        # (Kaisa) Move Looks end

        # (Kaisa) split "mesh" to "visuals" and "collisions" begin
        # 1. copy "mesh" prim
        # 2. rename two prims
        mesh_prim = stage.GetPrimAtPath(f"{geom_prim.GetPath().pathString}/mesh")
        if mesh_prim.IsValid():
            result1 = omni.kit.commands.execute("MovePrim",
            path_from=mesh_prim.GetPath(),
            path_to=f"{geom_prim.GetPath().pathString}/visuals"
            )
            # print(f"DEBUG visuals: {result1}")
            # print(f"DEBUG visuals path: {mesh_prim.GetPath()}")
            result2 = omni.kit.commands.execute("CopyPrim",
            path_from=f"{geom_prim.GetPath().pathString}/visuals",
            path_to=f"{geom_prim.GetPath().pathString}/collisions"
            )
            # print(f"DEBUG collisions: {result2}")
        # 3. change some attributes 
        collisions_mesh_prim = stage.GetPrimAtPath(f"{geom_prim.GetPath().pathString}/collisions")
        if collisions_mesh_prim.IsValid():
            # print("DEBUG dir prime: ", dir(collisions_mesh_prim))
            # 移除材质绑定
            if collisions_mesh_prim.HasRelationship("material:binding"):
                collisions_mesh_prim.GetRelationship("material:binding").ClearTargets(removeSpec=True)
                UsdGeom.Imageable(collisions_mesh_prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
                # print(f"已移除材质绑定: {collisions_mesh_prim}")
        # (Kaisa) split "mesh" to "visuals" and "collisions" begin

        # Move all meshes to underneath new Xform
        for child_mesh_prim in geom_prim.GetChildren():
            # (Kaisa) only add collider in "collisions" begin
            # if child_mesh_prim.GetTypeName() == "Mesh":
            if child_mesh_prim.GetName() == "collisions":
            # (Kaisa) only add collider in "collisions" begin
                # Apply collider properties to mesh
                if cfg.collision_props is not None:
                    # -- Collision approximation to mesh
                    # TODO: Move this to a new Schema: https://github.com/isaac-orbit/IsaacLab/issues/163
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_mesh_prim)
                    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
                    # -- Collider properties such as offset, scale, etc.
                    schemas.define_collision_properties(
                        prim_path=child_mesh_prim.GetPath(), cfg=cfg.collision_props, stage=stage
                    )
        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)
        # Apply default Xform rotation to mesh -> enable to set rotation and scale
        omni.kit.commands.execute(
            "CreateDefaultXformOnPrimCommand",
            prim_path=xform_prim.GetPath(),
            **{"stage": stage},
        )

        # Apply translation, rotation, and scale to the Xform
        geom_xform = UsdGeom.Xform(geom_prim)
        geom_xform.ClearXformOpOrder()

        # Remove any existing rotation attributes
        rotate_attr = geom_prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr:
            geom_prim.RemoveProperty(rotate_attr.GetName())

        # translation
        translate_op = geom_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(*cfg.translation))
        # rotation
        orient_op = geom_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        orient_op.Set(Gf.Quatd(*cfg.rotation))
        # scale
        scale_op = geom_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(*cfg.scale))

        # Handle instanceable
        # Create a new Xform prim that will be the prototype prim
        if cfg.make_instanceable:
            # Export Xform to a file so we can reference it from all instances
            export_prim_to_file(
                path=os.path.join(self.usd_dir, self.usd_instanceable_meshes_path),
                source_prim_path=geom_prim.GetPath(),
                stage=stage,
            )
            # Delete the original prim that will now be a reference
            geom_prim_path = geom_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[geom_prim_path], stage=stage)
            # Update references to exported Xform and make it instanceable
            geom_undef_prim = stage.DefinePrim(geom_prim_path)
            geom_undef_prim.GetReferences().AddReference(self.usd_instanceable_meshes_path, primPath=geom_prim_path)
            geom_undef_prim.SetInstanceable(True)

        # Apply mass and rigid body properties after everything else
        # Properties are applied to the top level prim to avoid the case where all instances of this
        #   asset unintentionally share the same rigid body properties
        # apply mass properties
        if cfg.mass_props is not None:
            # schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
            # maobo add  for add rigid_props  on geom_prim begin 
            schemas.define_mass_properties(prim_path=geom_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
            print(f"DEBUG cfg.mass_props: {cfg.mass_props}, type: {type(cfg.mass_props)}")
        # (Kaisa) get mass from 质量.txt begin
        # 此处默认文件与.obj在同一文件夹下
        else:
            # 尝试从文件读取质量
            try:
                mass_file = os.path.join(os.path.dirname(cfg.asset_path), "质量.txt")
                print("DEBUG insert 质量",{mass_file})
                if os.path.exists(mass_file):
                    print("DEBUG insert 质量文件存在")
                    with open(mass_file, 'r') as f:
                        content = f.read().strip()
                    mass_str = content.lower().replace('kg', '').strip()
                    mass = float(mass_str)
                    print("DEBUG mass = ",{mass})
                    cfg.mass_props = schemas_cfg.MassPropertiesCfg(mass=mass)
                    cfg.rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
                    schemas.define_mass_properties(prim_path=geom_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
                    print(f"从文件读取质量: {mass} kg")
            except:
                print("未设置质量")
        # (Kaisa) get mass from 质量.txt end 
        # apply rigid body properties
        if cfg.rigid_props is not None:
            # maobo add  for add rigid_props  on geom_prim end
            # schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)
            schemas.define_rigid_body_properties(prim_path=geom_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)
        # 修改材质相对路径,
        # (Kaisa) 添加更多纹理映射 begin
        for prim in stage.Traverse():
            if prim.GetName() == "Looks":
                for child_mesh_prim in prim.GetChildren():
                    if child_mesh_prim.GetTypeName() == "Material":
                        # print(child_mesh_prim)
                        material = UsdShade.Material(child_mesh_prim)
                        shaders = self.get_all_shaders_from_material(material)
                        for shader in shaders :
                            # 获取绑定的 Surface Shader
                            shader_prim = stage.GetPrimAtPath(shader['path'])
                            shader = UsdShade.Shader(shader_prim)
                            # (Kaisa) 完成路径更新和收集现有输入
                            material_folder_name = "materials"
                            usd_dir = os.path.dirname(self.usd_path)
                            textures_absolute_dir = os.path.join(usd_dir, material_folder_name)
                            existing_inputs = set()
                            for input_attr in shader.GetInputs():
                                # print(f"DEBUG shader output dir: {dir(input_attr)}")
                                input_name = input_attr.GetBaseName()
                                existing_inputs.add(input_name)
                                if input_attr.GetBaseName() == 'diffuse_texture':
                                    # print(f"DEBUG diffuse_texture exist! ")
                                    value = str(input_attr.Get())
                                    cleaned_path  = value.strip("@").split("textures/", 1)[-1]
                                    input_attr.Set("./textures/"+cleaned_path)
                            
                            # 2. 定义要创建的纹理输入映射
                            texture_inputs = {
                                'normalmap_texture': './' + material_folder_name + '/normal.jpg',
                                'reflectionroughness_texture': './' + material_folder_name + '/roughness.jpg',
                                'metallic_texture': './' + material_folder_name + '/metalness.jpg',
                                'ao_texture': './' + material_folder_name + '/Ambient Occlusion Map.jpg'
                            }
                            
                            # 3. 创建缺失的纹理输入
                            for input_name, texture_relative_path in texture_inputs.items():
                                if input_name not in existing_inputs:  # 直接使用已收集的集合
                                    # 检查文件是否存在
                                    texture_filename = texture_relative_path.replace('./' + material_folder_name + '/', '')
                                    absolute_texture_path = os.path.join(textures_absolute_dir, texture_filename)
                                    if os.path.exists(absolute_texture_path):
                                        # 创建新的纹理输入
                                        # print(f"DEBUG check texture absolute path: {absolute_texture_path}")
                                        input_attr = shader.CreateInput(input_name, Sdf.ValueTypeNames.Asset)
                                        input_attr.Set(texture_relative_path)
                                        print(f"创建新输入: {input_name} = {texture_relative_path}")
                                    else:
                                        print(f"纹理文件不存在: {absolute_texture_path}")
                                else:
                                    print(f"输入已存在: {input_name}")
                                    # (Kaisa) 添加更多纹理映射 end

        # Save changes to USD stage
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)
        

    def get_all_shaders_from_material(self, material):
        """
        获取材质中的所有shader
        
        参数:
        material: UsdShade.Material对象
        
        返回:
        shaders: 包含所有shader信息的字典列表
    """
        shaders = []
        
        # 遍历材质的所有子节点
        for child in material.GetPrim().GetChildren():
            if child.GetTypeName() == "Shader":
                shader = UsdShade.Shader(child)
                shader_info = {
                    "name": child.GetName(),
                    "type": child.GetTypeName(),
                    "path": child.GetPath().pathString,
                    "parameters": {}
                }
                shaders.append(shader_info)
                
        return shaders

    # (Kaisa) 自顶向下重命名 begin
    def safe_hierarchical_rename(self, stage, looks_prim_path):
        """安全的层级重命名"""
        looks_prim = stage.GetPrimAtPath(looks_prim_path)
        if not looks_prim.IsValid():
            return
        
        def rename_recursive(prim):
            prim_type = prim.GetTypeName()
            
            # 重命名当前 prim
            if prim_type == "Material" and prim.GetName() != "material":
                old_path = prim.GetPath().pathString
                parent_path = prim.GetPath().GetParentPath().pathString
                new_path = f"{parent_path}/material"
                
                print(f"重命名 Material: {old_path} -> {new_path}")
                omni.kit.commands.execute("MovePrim", path_from=old_path, path_to=new_path)
                
                # 重新获取重命名后的 prim
                prim = stage.GetPrimAtPath(new_path)
                
            elif prim_type == "Shader" and prim.GetName() != "Shader":
                old_path = prim.GetPath().pathString
                parent_path = prim.GetPath().GetParentPath().pathString
                new_path = f"{parent_path}/Shader"
                
                print(f"重命名 Shader: {old_path} -> {new_path}")
                omni.kit.commands.execute("MovePrim", path_from=old_path, path_to=new_path)
                
                # 重新获取重命名后的 prim
                prim = stage.GetPrimAtPath(new_path)
            
            # 递归处理子 prim（使用最新的 prim 对象）
            for child in prim.GetChildren():
                rename_recursive(child)
        
        # 开始递归
        rename_recursive(looks_prim)
        # (Kaisa) 自顶向下重命名 end

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(in_file: str, out_file: str, load_materials: bool = True) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to cm.

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        enable_extension("omni.kit.asset_converter")

        import omni.kit.asset_converter
        import omni.usd

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = True
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        task = instance.create_converter_task(in_file, out_file, None, converter_context)
        # Start conversion task and wait for it to finish
        success = await task.wait_until_finished()
        if not success:
            raise RuntimeError(f"Failed to convert {in_file} to USD. Error: {task.get_error_message()}")
        return success
