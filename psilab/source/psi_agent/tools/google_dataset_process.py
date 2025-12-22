import os
import zipfile
import shutil
import re


def list_top_level_directories(folder_path):
    # List all items in the folder
    items = os.listdir(folder_path)

    # Filter out the directories (items that are directories)
    directories = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

    # Print or return the directories
    for directory in directories:
        metadata_name = directory +"/metadata.txt"
        metadata_path = os.path.join(folder_path, metadata_name)
        print("===========================")
        print(metadata_name)
        category_name = extract_category_first_value(metadata_path)
        if(category_name is None or category_name ==""):
            category_name = "other"
        try:
            destination_folder="/home/zhwang/Albert/dataset//categories_folder/"+category_name
            source_folder = os.path.join(folder_path, directory)
            
            shutil.copytree(source_folder, os.path.join(destination_folder, os.path.basename(source_folder)))
            # shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
            print(f"Copied folder: {source_folder} -> {destination_folder}")
        except Exception as e:
            print(f"Error occurred: {e}")

def check_and_copy_to_categories_folder(root, categories_folder_name):
    # Define the path for categories folder
    
    print("+++++++++++++++++++++++++++++++_______________________")
    print(categories_folder_name)
    categories_folder_path = os.path.join('/home/zhwang/Albert/dataset/categories_folder', categories_folder_name)
    
    print(categories_folder_path)
    # Check if the categories folder exists
    if not os.path.exists(categories_folder_path):
        # If not, create it
        os.makedirs(categories_folder_path)
        print(f"Created directory: {categories_folder_path}")
    else:
        print(f"Directory already exists: {categories_folder_path}")

#  解析文件前  将 pbtxt  修改成 txt 。只需解析最后一个词 所以简化  Albert 
def rename_metadata_files(directory):
    categories_folder_name = ''
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is named 'metadata.pbtxt'
            if file == 'metadata.pbtxt':
                # Construct full file path
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, 'metadata.txt')
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} -> {new_path}')
                categories_folder_name = extract_category_first_value(new_path)
                if categories_folder_name is None or categories_folder_name == "":
                    categories_folder_name="other"
                check_and_copy_to_categories_folder(root ,categories_folder_name)
            if file =='metadata.txt':
                old_path = os.path.join(root, file)
                categories_folder_name = extract_category_first_value(old_path)
                if categories_folder_name is None or categories_folder_name == "":
                    categories_folder_name="other"
                check_and_copy_to_categories_folder(root ,categories_folder_name)
                
            
        
def extract_category_first_value(file_path):
    # Read the file contents
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expression to find the first category value and remove spaces
    match = re.search(r'categories\s*{[^}]*first\s*:\s*"([^"]+)"', content)

    if match:
        return match.group(1).replace(" ", "")  # Remove spaces
    else:
        return None
#  处理文件夹下的文件格局 copy texture.png 并删除不必要的信息 
def process_folders(root_dir):  
    # 遍历指定根目录下的所有文件夹
    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        meshes_path = os.path.join(folder_path, 'meshes')
        texture_path = os.path.join(folder_path,'materials/textures')
        print(texture_path)
        # 判断是否是文件夹
        if os.path.isdir(texture_path):
            print(f"Processing folder: {foldername}")
            
            # 查找 texture.png 文件并复制到 meshes 文件夹
            texture_path = os.path.join(texture_path, 'texture.png')
            if os.path.exists(texture_path):
                # 确保目标 meshes 文件夹存在
                os.makedirs(meshes_path, exist_ok=True)
                
                
                # 复制 texture.png 到 meshes 文件夹
                shutil.copy(texture_path, meshes_path)
                print(f"Copied texture.png from {foldername} to {meshes_path}")
            else:
                print(f"texture.png not found in {foldername}")
            
            # 删除 materials 文件夹
            materials_folder = os.path.join(folder_path, 'materials')
            if os.path.exists(materials_folder):
                shutil.rmtree(materials_folder)
                print(f"Deleted 'materials' folder in {foldername}")
            
            # 删除 model.config 和 model.sdf 文件
            for filename in ['model.config', 'model.sdf']:
                file_path = os.path.join(folder_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted {filename} in {foldername}")

def unzip_all(zip_dir):
    # 遍历目录中的所有文件
    for filename in os.listdir(zip_dir):
        # 判断是否为 zip 文件
        if filename.endswith(".zip"):
            zip_path = os.path.join(zip_dir, filename)
            
            # 创建一个新的文件夹以存放解压后的文件，文件夹名字与 zip 文件相同
            folder_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(zip_dir, folder_name)
            os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在，则创建
            
            # 解压文件
            print(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)  # 解压到指定的文件夹

            print(f"解压完成: {filename} -> {output_dir}")
            # 删除zip 文件
            os.remove(zip_path)

if __name__ == "__main__":
    # 指定包含 zip 文件的文件夹路径
    directory = "/home/zhwang/Albert/dataset/test"
    
    # file_path = "/home/zhwang/Albert/psi-isaaclab/source/psi_agent/psi_google_obejct_test/2_of_Jenga_Classic_Game/metadata.txt"
    
    # unzip_all(directory)
    # process_folders(directory)
    # rename_metadata_files(directory)
    list_top_level_directories(directory)

    print("finish")
   
