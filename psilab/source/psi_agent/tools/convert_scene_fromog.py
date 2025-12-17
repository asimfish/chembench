import cv2
import matplotlib.pyplot as plt
# import omni.usd
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdUtils
import os
import shutil
from pxr import Usd, UsdShade



def copy_texture_file(src_path, base_folder="data", target_root="/home/zhwang/Downloads/convert_og/chaoshi/cahoshi22/textures"):
    """
    复制纹理文件到当前目录的 `Textures/og_dataset/...` 目录，保持原有文件夹结构。

    :param src_path: 原始文件的完整路径
    :param base_folder: 需要截取的起始文件夹名称
    :param target_root: 目标根目录
    """
    # 规范化路径，去除首尾 `@`
    src_path = src_path.strip("@")

    # 确保文件存在
    if not os.path.isfile(src_path):
        print(f"文件不存在: {src_path}")
        return

    # 找到 `og_dataset` 在路径中的索引位置
    parts = src_path.split(os.sep)
    if base_folder in parts:
        base_index = parts.index(base_folder)  # 找到 `og_dataset` 的索引
        relative_path = os.sep.join(parts[base_index:])  # 截取 `og_dataset` 及其后面的路径
    else:
        print(f"未找到 `{base_folder}`，跳过: {src_path}")
        return

    # 目标文件路径
    target_path = os.path.join(target_root, relative_path)

    # 创建目标文件夹
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)

    # 复制文件
    try:
        shutil.copy(src_path, target_path)
        print(f"复制成功: {src_path} -> {target_path}")
    except Exception as e:
        print(f"复制失败: {src_path} -> {target_path}, 错误: {e}")
        
    return  relative_path
    

def update_shader_paths(usd_file):
    # 加载 USD Stage
    stage = Usd.Stage.Open(usd_file)
    # 遍历所有 prim
    for prim in stage.Traverse():
        # 检查 prim 是否是 Looks 组
        if prim.GetName() == "Looks":
            print(f"Found Looks prim at {prim.GetPath()}")
            
            # 遍历 Looks 下的所有 shader
            for material_prim in prim.GetChildren():
                if material_prim.GetTypeName() == "Material":
                    print(material_prim)
                    material = UsdShade.Material(material_prim)
                    shader_path = str(material.GetPath())+"/Shader"
                    print(shader_path)
            # 获取绑定的 Surface Shader
                    shader_prim = stage.GetPrimAtPath(shader_path)
                    print(shader_prim)
                    shader = UsdShade.Shader(shader_prim)
                    for input_attr in shader.GetInputs():
                            value = str(input_attr.Get())
                            print(f"Shader Input: {input_attr.GetBaseName()} = {value}")
                            if "/home/zhwang/Albert" in value:
                                print(f" Found Texture: {value}")
                                
                                relate_path = copy_texture_file(value) 
                                input_attr.Set("./textures/"+relate_path)
                                 
            stage.GetRootLayer().Save()
scene_path = "/home/zhwang/Downloads/convert_og/chaoshi/cahoshi22/chaoshi22.usd"
update_shader_paths(scene_path)

