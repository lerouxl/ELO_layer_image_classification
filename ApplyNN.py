from pathlib import Path
import argparse
import plotly
import os.path
from src import model as m
from src import image as im
from src import post_processing as pp
from src import graph
from train import ImageClassifier
import numpy as np

def loadmodel(modelpath):
    # Load the best model
    model = ImageClassifier(checkpoints_dir=r"checkpoints/TempApplyNN") # create object model and setup a temp folder to save in running data
    model.load_from_checkpoint(modelpath)
    return model

def class_repartition(df,img_path) :
    # Generate the class repartition chart
    pie_chart = graph.create_pie(df)
    pie_chart.write_html(f"{Path(img_path).stem}_Classification_chart.html")
    plotly.offline.plot(pie_chart, auto_play=False, filename=f"{Path(img_path).stem}_Classification_chart.html")

def Area_classification(df, img_path, score, width, height, CellSize) :
    image_classification_graph = graph.create_image_classification(df, img_path, score, width, height, CellSize)
    # image_classification_graph.write_html(f"{Path(img_path).stem}_ELO_area_classification.html")
    plotly.offline.plot(image_classification_graph, auto_play=False, filename=f"{Path(img_path).stem}_ELO_area_classification.html")

def ApplyNN(model, img_path: str, width: float, height: float, CellSize, htmlreport):
    """
    Take a layer image and classify area of this images
    :param model: pytorch model to apply
    :param img_path: The layer images
    :param width: Physical width of the image in mm
    :param height: Physical height of the images in mm
    :Cellsize: (size of tiles in mm, size of tiles in pixels)
    :return DataFrame with results
    """
    transform = m.get_transform()
    # Load layer image
    print("Load image")
    image = im.get_image(img_path, width, height,CellSize) # Get image resized to the right number of pixels, to be split equally and in mm proportion 
    image_size = image.size
    # Pre process the layer images
    print("Extract sub-images")
    images, img_pos = im.extract_sub_images(image, CellSize[1])  # get sub images in pixels
    images_torch = im.to_torch(images, transform)
    # Make a prediction
    print("Start Predictions")
    prediction = m.predict(model, images_torch)
    # Post process the prediction
    print("Generate scores and classifications of sub-images")
    score = pp.generate_score(prediction, image_size, CellSize[1], img_pos)
    classification = pp.get_classification(score)
    #class_names = ["bulging", "corner", "good", "porous", "powder"]
    #class_names = ["good", "porous", "bulging", "edges", "powder"]
    class_names = [ "good", "porous", "bulging", "corner","powder"]
    df = pp.pandas_results(classification, score, class_names)

    if htmlreport :
        # Generate the class repartition chart
        print("Generate the class repartition html chart")
        class_repartition(df,img_path) 
        # Generate the main report
        print("Generate the html area classification")
        Area_classification(df, img_path, score, width, height, CellSize) 

    return df # return DataFrame with results