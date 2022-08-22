# ELO layer image classification
____
## Presentation
This script is made of the Manuela dashboard to evaluate the printing quality using monitoring ELO images.
Printing layer ELO images are sliced in sub images of 5mm by 5mm (125px by 125px) and then classified in 5 categories:
- *powder*: The classified area is unmelted powder.
- *corner*: The classified area is the border between powder and a part.
- *porous*: The classified area is a porous part, not enough energy was given to fully melt the powder.
- *good*: The classified area was well printed, not defect were detected.
- *bulging*: The classified area is bulging, too much energy was given during the printing.

Two report are generated, the `Classification_chart.html` and the `ELO_area_classification.html`.
The first one is showing the percentage of each class in a pie chart.
The second one is showing the sub image classification on top of the layer ELO image. 

**Note**: The ELO images quality have a huge influence on the classification. 
As a noisy images can create features similar to porosity or bulging area.

## Installation
This script had been tested on python 3.7, and no GPU support were added.

1. Clone this GitHub repository `git clone https://github.com/lerouxl/ELO_layer_image_classification.git`.
2. Virtual environment:
   1. Install python 3.7 if needed.
   2. Create your environment: `virtualenv venv`.
   3. Activate it: `source venv/bin/activate`.
   4. Install dependencies: `pip install -r requirements.txt`.


## Use
Using a command line, launch `main.py` to classify one layer images.
`main.py` takes 3 arguments:
- `--img` : The path to an ELO image.
- `--width`: The physical width of the image.
- `--height`: The physical height of the image.

`python main.py --img data/example.tiff --width 200 --height 200`
This will generate two results files:
- ***"image name"*_Classification_chart.html**: A pie chart showing the proportion of each class.
- ***"image name"*_ELO_area_classification.html**: The layer ELO images with transparent squares showing the area classification.

The size of the ELO_area_classification.html's classified tiles can be modified with the slider.
The user must update there size to fit their uses.

## Results example and screenshots:
Example html results files can be found at the root of this repo:
- [example_Classification_chart.html](https://github.com/lerouxl/ELO_layer_image_classification/blob/main/example_Classification_chart.html)
- [example_ELO_area_classification.html](https://github.com/lerouxl/ELO_layer_image_classification/blob/main/example_ELO_area_classification.html)

Here are some screenshots:

### example_Classification_chart.html screenshot
This pie chart represent the proportion of each class in the layer.
![alt text](https://github.com/lerouxl/ELO_layer_image_classification/blob/main/data/Classification_chart.png?raw=true)

### example_ELO_area_classification.html screenshots
The layer ELO images in background with transparent squares showing the area classification.
The square size can be adjusted by moving the cursor.
![alt text](https://github.com/lerouxl/ELO_layer_image_classification/blob/main/data/Area_classification_results.png?raw=true)
By clinking it's possible to zoom on the image.
Hovering the mouse on a square allow to see its classification name and the score of each class.
![alt text](https://github.com/lerouxl/ELO_layer_image_classification/blob/main/data/Area_classification_results_zoom.png?raw=true)





