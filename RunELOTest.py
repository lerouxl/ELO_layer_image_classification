from SplitImages import *
from ApplyNN import *
from train import *
import matplotlib.pyplot as plt
import numpy as np
from train import TrainNN
from random_delete_image import class_balancer

def SliceImages(GridSplit):
    print("Split Images")
    dataset = Path(r"data/ELO")
    outputfolder = Path(r"data/ELO_jpg_subimages_4_class")
    img_size = (352, 352)  # Size of images in dataset folder
    PBorder = 0.2  # Percentage of images containing the border on each side. 0.2 = 20%
    # validation_percent = 0.1 # between 0.00 and 1.00 - Original percentage
    validation_percent = 0.2  # between 0.00 and 1.00 - New percentage for testing smaller numebr of images
    BorderSize = int(PBorder*img_size[0])               # Border size in pixels
    CellSizePx = int((img_size[0] - 2*BorderSize)/GridSplit)  # Square Cells size in pixels 
    print("Image Size in px: ",img_size[0])
    print("Border Size in px:",BorderSize)
    print("Grid: ",GridSplit, "x",GridSplit)
    print("Cell Size in px: ",CellSizePx)
    SplitImages(CellSizePx,GridSplit,BorderSize, dataset,outputfolder,validation_percent) # Split using pixels
    # Info about images used in model training
    TrainImageSize = (25,25)    # Size of training images before grid split in mm, 
    CellSizemm=(1-2*PBorder)*TrainImageSize[0]/(GridSplit)  # Size of images in bulk (class 0,1 and 2) in mm used in training, this info should be stored in the model
    print("Cell Size in mm: ", CellSizemm)
    return (CellSizemm,CellSizePx) # return cell size in mm and pixels

def PerformTraining(CellSize,max_epochs, checkpoint_suffix="", learning_rate=0.001) :
    # train new model
    print("Train New Model")
    accelerator  = "gpu"  # "gpu" or "cpu": define if the training is done on the graphic card (faster) or the cpu
    TrainPath = r"./data/ELO_jpg_subimages_4_class/train"
    ValPath = r"./data/ELO_jpg_subimages_4_class/val"
    TestPath = r"./data/ELO_jpg_subimages_4_class/val"
    nb_classes = 5
    ModelPath = TrainNN(accelerator, TrainPath,  ValPath, TestPath, max_epochs, nb_classes, 
                        CellSize=CellSize, suffix=checkpoint_suffix,learning_rate=learning_rate) 
    # cellsize transfered to try to link it to the saved model, not working yet
    return ModelPath
 
def PerformPredicition(ModelPath, CellSize, imagepath,TestImageSize,htmlreport) :
    """Take the path to a ckpt model and load the "last_model_weight.pt" weight, and apply it to an image

    Parameters
    ----------
    ModelPath : Path
        Path to the model (it can be last of best.cktp, their weight will not be used due to a bug in save), the used weight will be their neighboor file last_model_weight.pt
    CellSize : _type_
        _description_
    imagepath : str or Path
        Path to the image to apply
    TestImageSize : Tuple
        Tuple of the image dimension
    htmlreport : bool
        True of False to indicate if we want an html report
    """
    print("Apply NN to image: ", imagepath)
    #TestImageSize = (200,200) # Size of new tested image in mm
    model = loadmodel(ModelPath)

    model.load_state_dict(torch.load(
        Path(ModelPath).parent / "last_model_weight.pt"))
    model.eval()

    prediction = ApplyNN(model,imagepath, TestImageSize[0], TestImageSize[1], CellSize,htmlreport)
    # Save results in a csv file
    print("Save Results to CSV")
    filename = imagepath+".csv"
    filename = filename.replace("/", "-")
    prediction.to_csv(filename, index=False)


if False: #__name__ == "__main__":
#if True: #__name__ == "__main__":
    GridSplit = 3  # 3 => 3x3 split in the bulk, the sides are removed and used to create powder class and corner class images, corner class should be named "edges"

    CellSize = (5.0, 70)
    #CellSize = SliceImages(GridSplit)
    # Make all class have the same number of images.
    #class_balancer(r"data\ELO_jpg_subimages_4_class\train")
    #class_balancer(r"data\ELO_jpg_subimages_4_class\val")
    print("CellSize: ", CellSize)

    #CellSize = (5,70) # (mm, pixels) # to test the end without re-doing a split
    max_epochs = 80 # Maximal number of training epoch
    learning_rate = 0.001 # 0.001 in the ELO image classification paper
    ModelPath = PerformTraining(CellSize,max_epochs,checkpoint_suffix=f"_grid_split_{GridSplit}") 
    TestImageSize = (25,25) # Size of new tested image in mm

    print(f"Apply the best model : {ModelPath}")
    PerformPredicition(ModelPath, CellSize, "data/11-03-44_09.jpg", TestImageSize, htmlreport=True) 
 
    # used in terminal to check confusion matrix: tensorboard --logdir .\checkpoints\

if True: #__name__ == "__main__":
#if False: #__name__ == "__main__":
    ModelPath = r"D:\Documents\Users Files\Samuel\ELO_layer_image_classification v2\checkpoints\2023_02_16_18_32_44_grid_split_3\last.ckpt"   
    CellSize = (5.0, 70)
    TestImageSize = (25,25)
    PerformPredicition(ModelPath, CellSize, "10-26-09_02.jpg", TestImageSize, htmlreport=True) 
    print("Finished!!")