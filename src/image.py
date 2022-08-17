import torchvision.transforms as T
import torch
from PIL import Image
import cv2
import numpy as np


def get_image(path: str, width: float, height: float) -> Image:
    """
    Load one image and return it as a PIL image.
    :param path: Path to the image to load.
    :param width: Physical width of the image in mm.
    :param height: Physical height of the image in mm.
    :return: The PIL Image
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)  # uint8 image
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Images used in training had an physical dimension of 5 by 5 mm
    # 125 px -> 5 mm
    width_p, height_p, _ = image.shape
    sp = (width_p / width) * 5
    scale_width = 125 / sp

    sp = (height_p / height) * 5
    scale_height = 125 / sp

    image = cv2.resize(image, dsize=( int(width_p*scale_width),
                                      int(height_p * scale_height)))

    # Rescale it to be divisible by 125 (size of the chunck)
    width_p, height_p,_ = image.shape
    target_w = int(round(width_p/125)) * 125
    target_h = int(round(height_p/125)) * 125

    image = cv2.resize(image,dsize=(target_w,target_h))

    image = Image.fromarray((image * 255).astype(np.uint8))

    return image


def extract_sub_images(image: Image, crop_size=(125, 125)):
    """
    Extract sub images form a PIL image and return them in a list.
    :param image: PIL images to extract crop.
    :param crop_size: tuple, x and y size of the sub images.
    :return: 2 list. The list of the images and there position on the images
    """
    image_size = image.size

    top = 0
    images = []
    img_pos = []
    i = 0
    for bottom in range(crop_size[0], image_size[0] + crop_size[0], crop_size[0]):
        left = 0
        j = 0
        for right in range(crop_size[1], image_size[1] + crop_size[1], crop_size[1]):
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
