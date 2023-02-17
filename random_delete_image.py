r"""
This code is made to DELETE images from a dataset, so have a backup of your data before using this code.

This code will look list the number of images per class folder and remove random images until each class have the same number of sample.

Expected folder architecture:

    Folder:
        class_1:
            1.jpg
            2.jpg
            ....jpg
            1000.jpg
        class_2:
            1.jpg
            2.jpg
            ....jpg
            1100.jpg
        class_3:
            1.jpg
            2.jpg
            ....jpg
            4100.jpg

Expected output:
    Folder:
        class_1:
            1.jpg
            2.jpg
            ....jpg
            1000.jpg
        class_2:
            1.jpg
            2.jpg
            ....jpg
            1000.jpg
        class_3:
            1.jpg
            2.jpg
            ....jpg
            1000.jpg
"""
from pathlib import Path
import random

def class_balancer(folder_path):
    folder_path = Path(folder_path)
    # List every class folder in this folder:
    class_folders = []

    for path in folder_path.iterdir():
        if path.is_dir():
            class_folders.append(path)

    print(f"In {folder_path.name}, {len(class_folders)} class folder have been found")

    # Check the number of images per class
    class_size = []
    for class_f in class_folders:
        class_size.append(len(list(class_f.glob("*.jpg"))))

    print(f"The number of images per class is: {class_size}")
    print(f"The percentage of images per class is: { [ f'{int(100* c / sum(class_size))}%' for c in class_size]  }")

    sample_size = min(class_size)
    print(f"The sample size selected is {sample_size}")

    # Remove unwanted files:

    for class_folder, actual_size in zip(class_folders, class_size):
        if actual_size == sample_size:
            print(f"{class_folder.name} already have the selected number of files")
            continue # Go to the next iteration
        else:
            nb_to_del = actual_size - sample_size
            print(f"{class_folder.name} have {actual_size} images, the actual target is {sample_size}, {nb_to_del} will be delected")
            # list all images
            class_images = list(class_folder.glob("*.jpg"))
            # Select nb_to_del images randomly
            to_del = random.sample(class_images, nb_to_del)

            for file in to_del:
                file.unlink() # Remove file


class_balancer(Path(r"data\sampled_ELO_4_class\train"))
class_balancer(Path(r"data\sampled_ELO_4_class\val"))