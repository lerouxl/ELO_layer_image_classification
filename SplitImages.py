from pathlib import Path
from PIL import Image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random

def normalise(img : np.ndarray) -> np.ndarray: 
    """Take one image and normalise it with value between 0 and 255"""
    shape = img.shape
    normalizedImg = np.zeros(shape)
    normalizedImg = cv.normalize(img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    
    return normalizedImg

def display_cv2(img: np.ndarray, title: str = "") -> None:
    """Display a cv2 image as a plt element"""
    img2 = img[:,:,::-1]
    plt.imshow(img2)
    plt.title(title)
    plt.show()

def process_files_with_crops(img_in, step, img_size, BorderSize, folder) -> None: 
    """Process one file img_in and save it to img_out"""
    # Load and normalise data
    img = cv.imread(str(img_in))
    img = normalise(img)
    img = cv.resize(img, img_size)  
    x_max, y_max,_ = img.shape
    # Create split in    
    if int(x_max/BorderSize) < x_max/BorderSize :
        x_max = x_max - BorderSize
    if int(y_max/BorderSize) < y_max/BorderSize :
        y_max = y_max - BorderSize
    # change xmax and ymax if bigger than x times BorderSize
    past_x = 0
    for x in range(BorderSize,x_max + BorderSize,BorderSize): # Select the pixel position of the right side of each split from BorderSize to x_max
        past_y = 0
        for y in range(BorderSize,y_max + BorderSize,BorderSize): # Select the pixel position of the bottom side of each split from BorderSize to y_max
            crop = img[past_x:x, past_y:y]          
            if past_x == 0 or past_y == 0 or x == x_max or y == y_max: # select only the border
                categorie = str(3)
                name = f"{Path(img_in).stem}_{x}_{y}.jpg"
                save_name = Path(folder) / Path(categorie) /  Path(name) 
                save_name.parent.mkdir(parents=True, exist_ok=True)
                save_name = str(save_name)
                cv.imwrite(save_name, crop)
                # Create powder bed images class
                # Do a vertical miror
                if past_y == 0 :
                    powder = np.hstack([crop[0:BorderSize, 0:BorderSize//2], np.fliplr( crop[0:BorderSize, 0:BorderSize//2])])
                elif y == y_max:
                    powder = np.hstack([crop[0:BorderSize, BorderSize//2: BorderSize], np.fliplr(crop[0:BorderSize, BorderSize//2: BorderSize])]) 
                # Do a horizontal mirror
                elif past_x == 0:
                    powder = np.vstack([crop[0:BorderSize//2, 0:BorderSize], np.flipud( crop[0:BorderSize//2, 0:BorderSize])])
                elif past_x == x_max:
                    powder = np.vstack([crop[BorderSize//2:BorderSize, 0:BorderSize], np.flipud( crop[BorderSize//2:BorderSize, 0:BorderSize])])                                                   
                                                                                                            
                save_name = Path(folder) /"4" /  Path(name) # Different
                save_name.parent.mkdir(parents=True, exist_ok=True)
                save_name = str(save_name)
                cv.imwrite(save_name, powder)
            else:
                pass                   
            past_y = y
        past_x = x
# Create split in the bulk  
    x_max, y_max,_ = img.shape     
    past_x = BorderSize
    for x in range(BorderSize + step,x_max + step-BorderSize,step): # Select the pixel position of the right side of each split from step to x_max
        past_y = BorderSize
        for y in range(BorderSize + step,y_max+ step-BorderSize,step): # Select the pixel position of the bottom side of each split from step to y_max
            crop = img[past_x:x, past_y:y]          
            categorie = Path(img_in).parent.name
            
            name = f"{Path(img_in).stem}_{x}_{y}.jpg"
            save_name = Path(folder) / Path(categorie) /  Path(name) 
            save_name.parent.mkdir(parents=True, exist_ok=True)
            save_name = str(save_name)
            cv.imwrite(save_name, crop)     

            past_y = y
        past_x = x
# SplitImages(CellSize,GridSplit,BorderSize, dataset,outputfolder,validation_percent) # Split using pixels
                  
def SplitImages(step,GridSplit,BorderSize,dataset,outputfolder,validation_percent) :
    img_size = (2*BorderSize+step*GridSplit,2*BorderSize+step*GridSplit) 
    all_tif = dataset.rglob("*.tif") # list of files paths (including location and names)
    all_tif = list(all_tif)
    random.shuffle(all_tif)

    number_of_tif = len(all_tif)
    number_validation = int(validation_percent*number_of_tif)
    number_train = number_of_tif - number_validation
    print(f"There are {number_of_tif} images, {number_validation} images will be used for validation and {number_train} for training")

    # Create where the final dataset will be stored
    outputfolder.mkdir(parents=True, exist_ok=True)
    (outputfolder / "train").mkdir(parents=True, exist_ok=True)
    (outputfolder / "val").mkdir(parents=True, exist_ok=True)

    #Split files between the train and validation dataset
    to_validation = all_tif[0:number_validation]  # take paths of validation images
    to_train = all_tif[number_validation:number_train] # take paths of training images
    len(to_validation), len(to_train)

    #Create small images
    print(f"Creating Training in folder ", outputfolder)
    for i in tqdm(to_train):
        process_files_with_crops(i,step,img_size,BorderSize, outputfolder / "train")
    print(f"Creating Validation Images")
    for i in tqdm(to_validation):
        process_files_with_crops(i,step,img_size,BorderSize, outputfolder / "val")
