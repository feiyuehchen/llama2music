import os

def traverse_dir(root_dir,
                extension, # extension=('mp3', 'wav')
                is_sort = False, 
                is_pure = False
                ):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                file_list.append(pure_path)
    
    if is_sort:
        print("sort the file list")
        file_list.sort()
    print(f"total count of the file: {len(file_list)}")
    return file_list