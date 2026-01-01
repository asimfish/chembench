"""
检查 USD 物体的材质属性（材质类型、Opacity 等）
可以在 Isaac Sim 中运行，或者独立加载 USD 文件检查

用法：
1. 在 Isaac Sim 中运行（场景已加载）:
   python check_material.py --prim_path /World/envs/env_0/Bottle
   
2. 直接检查 USD 文件:
   python check_material.py --usd_path /path/to/asset.usd
"""

import argparse
from pxr import Usd, UsdShade, UsdGeom, Sdf


def get_material_info(stage, prim_path: str) -> dict:
    """
    获取指定 prim 的材质信息
    
    Returns:
        dict: 包含材质类型、shader 类型、opacity 等信息
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {"error": f"Prim not found: {prim_path}"}
    
    result = {
        "prim_path": prim_path,
        "prim_type": prim.GetTypeName(),
        "materials": []
    }
    
    # 获取所有绑定的材质
    binding_api = UsdShade.MaterialBindingAPI(prim)
    
    # 直接绑定的材质
    direct_binding = binding_api.GetDirectBinding()
    if direct_binding.GetMaterial():
        material_info = analyze_material(stage, direct_binding.GetMaterial())
        material_info["binding_type"] = "direct"
        result["materials"].append(material_info)
    
    # 遍历子 prim 查找更多材质
    for child in prim.GetAllChildren():
        child_binding = UsdShade.MaterialBindingAPI(child)
        child_direct = child_binding.GetDirectBinding()
        if child_direct.GetMaterial():
            mat = child_direct.GetMaterial()
            # 避免重复
            if not any(m.get("material_path") == str(mat.GetPath()) for m in result["materials"]):
                material_info = analyze_material(stage, mat)
                material_info["binding_type"] = f"child ({child.GetPath()})"
                result["materials"].append(material_info)
    
    # 如果没有找到材质，尝试在整个层级中搜索
    if not result["materials"]:
        result["materials"] = find_materials_in_hierarchy(stage, prim)
    
    return result


def analyze_material(stage, material: UsdShade.Material) -> dict:
    """分析单个材质的属性"""
    mat_info = {
        "material_path": str(material.GetPath()),
        "material_name": material.GetPrim().GetName(),
        "shaders": []
    }
    
    # 获取 Surface Output
    surface_output = material.GetSurfaceOutput()
    if surface_output:
        connected_source = surface_output.GetConnectedSource()
        if connected_source and connected_source[0]:
            shader_prim = connected_source[0].GetPrim()
            shader_info = analyze_shader(shader_prim)
            shader_info["connection"] = "surface"
            mat_info["shaders"].append(shader_info)
    
    # 遍历材质下的所有 Shader
    for child in material.GetPrim().GetAllChildren():
        if child.GetTypeName() == "Shader":
            # 避免重复
            if not any(s.get("shader_path") == str(child.GetPath()) for s in mat_info["shaders"]):
                shader_info = analyze_shader(child)
                mat_info["shaders"].append(shader_info)
    
    return mat_info


def analyze_shader(shader_prim) -> dict:
    """分析 Shader 的属性"""
    shader = UsdShade.Shader(shader_prim)
    
    # 获取 Shader ID（确定材质类型）
    shader_id = shader.GetShaderId()
    
    shader_info = {
        "shader_path": str(shader_prim.GetPath()),
        "shader_name": shader_prim.GetName(),
        "shader_id": shader_id if shader_id else "Unknown",
        "shader_type": classify_shader_type(shader_id),
        "inputs": {}
    }
    
    # 获取所有输入参数
    for input_attr in shader.GetInputs():
        input_name = input_attr.GetBaseName()
        input_value = input_attr.Get()
        
        # 特别关注透明度相关参数
        if input_value is not None:
            shader_info["inputs"][input_name] = format_value(input_value)
    
    # 提取关键透明度参数
    opacity_params = extract_opacity_params(shader_info["inputs"])
    shader_info["opacity_summary"] = opacity_params
    
    return shader_info


def classify_shader_type(shader_id: str) -> str:
    """根据 Shader ID 分类材质类型"""
    if not shader_id:
        return "Unknown"
    
    shader_id_lower = shader_id.lower()
    
    # OmniPBR
    if "omnipbr" in shader_id_lower:
        return "OmniPBR"
    
    # OmniGlass
    if "omniglass" in shader_id_lower or "glass" in shader_id_lower:
        return "OmniGlass"
    
    # UsdPreviewSurface
    if "usdpreviewsurface" in shader_id_lower or "preview" in shader_id_lower:
        return "UsdPreviewSurface"
    
    # MDL 材质
    if ".mdl" in shader_id_lower:
        if "glass" in shader_id_lower:
            return "MDL Glass"
        elif "pbr" in shader_id_lower:
            return "MDL PBR"
        else:
            return f"MDL ({shader_id})"
    
    return f"Other ({shader_id})"


def extract_opacity_params(inputs: dict) -> dict:
    """提取透明度相关参数"""
    opacity_params = {}
    
    # 常见的透明度参数名
    opacity_keys = [
        "opacity", "opacity_constant", "opacity_texture",
        "alpha", "transparency", "transmission",
        "enable_opacity", "opacity_threshold",
        "glass_ior", "ior", "thin_walled",
        "cutout_opacity", "opacity_mode"
    ]
    
    for key in opacity_keys:
        for input_name, input_value in inputs.items():
            if key.lower() in input_name.lower():
                opacity_params[input_name] = input_value
    
    return opacity_params


def find_materials_in_hierarchy(stage, root_prim) -> list:
    """在层级中搜索所有材质"""
    materials = []
    
    for prim in Usd.PrimRange(root_prim):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct = binding_api.GetDirectBinding()
        if direct.GetMaterial():
            mat = direct.GetMaterial()
            if not any(m.get("material_path") == str(mat.GetPath()) for m in materials):
                mat_info = analyze_material(stage, mat)
                mat_info["bound_to"] = str(prim.GetPath())
                materials.append(mat_info)
    
    return materials


def format_value(value):
    """格式化值以便显示"""
    if isinstance(value, (tuple, list)):
        return [format_value(v) for v in value]
    elif hasattr(value, '__iter__') and not isinstance(value, str):
        try:
            return list(value)
        except:
            return str(value)
    else:
        return value


def print_material_info(info: dict, verbose: bool = False):
    """打印材质信息"""
    print("\n" + "=" * 70)
    print(f"🔍 Prim: {info.get('prim_path', 'Unknown')}")
    print(f"   Type: {info.get('prim_type', 'Unknown')}")
    print("=" * 70)
    
    if "error" in info:
        print(f"❌ Error: {info['error']}")
        return
    
    materials = info.get("materials", [])
    if not materials:
        print("⚠️  No materials found!")
        return
    
    for i, mat in enumerate(materials):
        print(f"\n📦 Material {i+1}: {mat.get('material_name', 'Unknown')}")
        print(f"   Path: {mat.get('material_path', 'N/A')}")
        if mat.get("binding_type"):
            print(f"   Binding: {mat['binding_type']}")
        if mat.get("bound_to"):
            print(f"   Bound to: {mat['bound_to']}")
        
        shaders = mat.get("shaders", [])
        for j, shader in enumerate(shaders):
            print(f"\n   🎨 Shader {j+1}: {shader.get('shader_name', 'Unknown')}")
            print(f"      Type: {shader.get('shader_type', 'Unknown')}")
            print(f"      ID: {shader.get('shader_id', 'N/A')}")
            
            # 透明度摘要
            opacity_summary = shader.get("opacity_summary", {})
            if opacity_summary:
                print(f"\n      📊 Opacity Parameters:")
                for key, value in opacity_summary.items():
                    print(f"         • {key}: {value}")
            else:
                print(f"\n      📊 No explicit opacity parameters found")
            
            # 详细输入参数
            if verbose:
                print(f"\n      📝 All Inputs:")
                for key, value in shader.get("inputs", {}).items():
                    print(f"         • {key}: {value}")
    
    print("\n" + "=" * 70)


def check_usd_file(usd_path: str, verbose: bool = False):
    """检查 USD 文件中的所有材质"""
    print(f"\n📂 Loading USD file: {usd_path}")
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"❌ Failed to open USD file: {usd_path}")
        return
    
    # 获取根 prim
    root = stage.GetPseudoRoot()
    
    # 收集所有材质
    all_materials = {}
    
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() == "Material":
            mat = UsdShade.Material(prim)
            mat_info = analyze_material(stage, mat)
            all_materials[str(prim.GetPath())] = mat_info
    
    print(f"\n🔎 Found {len(all_materials)} material(s) in USD file:")
    
    for mat_path, mat_info in all_materials.items():
        print(f"\n{'=' * 60}")
        print(f"📦 Material: {mat_info.get('material_name', 'Unknown')}")
        print(f"   Path: {mat_path}")
        
        for shader in mat_info.get("shaders", []):
            print(f"\n   🎨 Shader: {shader.get('shader_name', 'Unknown')}")
            print(f"      Type: {shader.get('shader_type', 'Unknown')}")
            print(f"      ID: {shader.get('shader_id', 'N/A')}")
            
            opacity_summary = shader.get("opacity_summary", {})
            if opacity_summary:
                print(f"\n      📊 Opacity Parameters:")
                for key, value in opacity_summary.items():
                    print(f"         • {key}: {value}")
            
            if verbose:
                print(f"\n      📝 All Inputs:")
                for key, value in shader.get("inputs", {}).items():
                    print(f"         • {key}: {value}")


def check_runtime_prim(prim_path: str, verbose: bool = False):
    """在运行时检查指定 prim 的材质"""
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if not stage:
            print("❌ No stage loaded in Isaac Sim!")
            return
        
        info = get_material_info(stage, prim_path)
        print_material_info(info, verbose)
        
    except ImportError:
        print("❌ omni.usd not available. Use --usd_path to check USD files directly.")


# ============================================================
# 常用 USD 资产路径
# ============================================================
ASSET_PATHS = {
    "glass_beaker_100ml": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/glass_beaker_100ml/Beaker003.usd",
    "mortar": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/mortar/Mortar001.usd",
    "brown_reagent_bottle": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/brown_reagent_bottle_large/ReagentBottle001.usd",
    "clear_reagent_bottle": "/home/psibot/chembench/psilab/assets/usd/asset_collection/sim_ready/solid_assets/clear_reagent_bottle_large/ReagentBottle002.usd",
}


def main():
    parser = argparse.ArgumentParser(description="Check USD material properties")
    parser.add_argument("--prim_path", type=str, help="Runtime prim path (e.g., /World/envs/env_0/Bottle)")
    parser.add_argument("--usd_path", type=str, help="Path to USD file")
    parser.add_argument("--asset", type=str, choices=list(ASSET_PATHS.keys()), help="Predefined asset name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all shader inputs")
    parser.add_argument("--list_assets", action="store_true", help="List predefined assets")
    
    args = parser.parse_args()
    
    if args.list_assets:
        print("\n📋 Predefined Assets:")
        for name, path in ASSET_PATHS.items():
            print(f"   • {name}: {path}")
        return
    
    if args.asset:
        args.usd_path = ASSET_PATHS[args.asset]
    
    if args.usd_path:
        check_usd_file(args.usd_path, args.verbose)
    elif args.prim_path:
        check_runtime_prim(args.prim_path, args.verbose)
    else:
        # 默认检查一些常用资产
        print("\n📋 Checking common assets...")
        for name, path in list(ASSET_PATHS.items())[:3]:
            print(f"\n{'#' * 60}")
            print(f"# Asset: {name}")
            print(f"{'#' * 60}")
            try:
                check_usd_file(path, args.verbose)
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

