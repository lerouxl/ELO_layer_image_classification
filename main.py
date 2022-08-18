from pathlib import Path
import argparse
import plotly
import os.path
from src import model as m
from src import image as im
from src import post_processing as pp
from src import graph


def main(img_path: str, width: float, height: float):
    """
    Take a layer image and classify area of this images
    :param img_path: The layer images
    :param width: Physical width of the image in mm
    :param height: Physical height of the images in mm
    :return:
    """
    # Load the model and the image transformation function
    model = m.get_model(r"model/image_classification_model.pt")
    transform = m.get_transform()

    # Load layer image
    image = im.get_image(img_path, width, height)
    image_size = image.size
    crop_size = (125, 125)

    # Pre process the layer images
    images, img_pos = im.extract_sub_images(image, crop_size)
    images_torch = im.to_torch(images, transform)

    # Make a prediction
    prediction = m.predict(model, images_torch)

    # Post process the prediction
    score = pp.generate_score(prediction, image_size, crop_size, img_pos)
    classification = pp.get_classification(score)

    class_names = ["bulging", "corner", "good", "porous", "powder"]
    df = pp.pandas_results(classification, score, class_names)

    # Generate the class repartition chart
    pie_chart = graph.create_pie(df)
    pie_chart.write_html(f"{Path(img_path).stem}_Classification_chart.html")
    plotly.offline.plot(pie_chart, auto_play=False,
                        filename=f"{Path(img_path).stem}_Classification_chart.html")

    # Generate the main report
    image_classification_graph = graph.create_image_classification(df, img_path, score, width, height)
    # image_classification_graph.write_html(f"{Path(img_path).stem}_ELO_area_classification.html")
    plotly.offline.plot(image_classification_graph, auto_play=False,
                        filename=f"{Path(img_path).stem}_ELO_area_classification.html")


def valid_file(string):
    if not os.path.exists(string):
        raise FileNotFoundError
    else:
        return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify powder layer.')
    parser.add_argument('-i', '--img', type=valid_file, help='Where is the layer images to classify.')
    parser.add_argument('--width', type=float, help='Physical width of the image in mm.')
    parser.add_argument('--height', type=float, help='Physical height of the image in mm.')

    args = parser.parse_args()

    # img = r"data/example.tif"
    # width = int(125)
    # height = int(125)
    main(img_path=args.img, width=args.width, height=args.height)
    # main(img_path=img, width=width, height=height)
