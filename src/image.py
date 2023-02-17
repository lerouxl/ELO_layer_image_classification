import torchvision.transforms as T
import torch
from PIL import Image
import cv2
import numpy as np

def get_image(path: str, width: float, height: float, CellSize) -> Image:
    """
    Load one image and return it as a PIL image.
    Get image resized to the right number of pixels, to be split equally and in mm proportion of cellsize
    :param path: Path to the image to load.
    :param width: Physical width of the image in mm.
    :param height: Physical height of the image in mm.
    :return: The Image in proportion of cellsize
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)  # uint8 image
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Rescale it so a pixel of the new image has the same size as what was used
    # and for it to be divisible by PixelW and PixelH (size of the cells)
    target_w = int(round(int(CellSize[1]*width/CellSize[0])/CellSize[1])) * CellSize[1]
    target_h = int(round(int(CellSize[1]*height/CellSize[0])/CellSize[1])) * CellSize[1]

    image = cv2.resize(image,dsize=(target_w,target_h))

    image = Image.fromarray((image * 255).astype(np.uint8)) # why 255?

    return image

def extract_sub_images(image: Image, CellSizePx):
    """
    Extract sub images form a PIL image and return them in a list.
    :param image: PIL images to extract crop.
    :param PixelW and PixelH: tuple, x and y size of the sub images.
    :return: 2 list. The list of the images and there position on the images
    """
    image_size = image.size

    top = 0
    images = []
    img_pos = []
    i = 0
    for bottom in range(CellSizePx, image_size[0] + CellSizePx, CellSizePx):
        left = 0
        j = 0
        for right in range(CellSizePx, image_size[1] + CellSizePx, CellSizePx):
            images.append(image.crop((left, top, right, bottom)))
            img_pos.append((i, j))
            left = right
            j = j + 1
        top = bottom
        i = i + 1

    return images, img_pos

def to_torch(images: list, transform: T.transforms.Compose) -> torch.Tensor:
    """
    Transform a list of images into a tensors
    :param images: list of PIL images
    :param transform: transformation funciton to apply
    :return: A tensor containing all the images
    """
    image = transform(images[0]).unsqueeze(0)

    for im in images[1:]:
        image = torch.vstack((image, transform(im).unsqueeze(0)))

    return image
