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
   3. Activate it: `source venv/bin/activate` or `.\venv\Scripts\activate` on windows.
   4. Install dependencies: `pip install -r requirements.txt`.

Or, use anaconda to create an environment (recomanded):
1. Clone this GitHub repository `git clone https://github.com/lerouxl/ELO_layer_image_classification.git`.
2. Create the conda environment environment:
   1. With anaconda promt, go the the code repository
   2. Create the environment: `conda env create -f environment.yml`
   3. Activate it: `conda activate ELO`
   4. Launch the classification or training code.
   

## Use
Using a command line, launch `main.py` to classify one layer images.
`main.py` takes 3 arguments:
- `--img` : The path to an ELO image.
- `--width`: The physical width of the image.
- `--height`: The physical height of the image.

`python main.py --img data/example.tif --width 200 --height 200`
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

____
## Train on new dataset

To generate a new AI model, the `train.py` script can be used.
This script require to have a dataset split in 2 folders (`train` and `val`) made of subfolder named with the class names of their pictures as describe bellow:
```
dataset_name
|_>train
|   |___>class_1
|   |___>class_2
|   |___>class_3
|   |___>class_4
|   |___>class_5 
|         |___> *.jpg
|         |___> *.jpg
|_>val
   |___>class_1
   |___>class_2
   |___>class_3
   |___>class_4
   |___>class_5 
         |___> *.jpg
         |___> *.jpg
```

Then the `train_dataloader` and `val_dataloader` must be updated to train the AI on the new dataset. For this, the image folder must be modified to be:

```python
def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder("./data/dataset_name/train", transform=self.transform)
        
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.hparams.batch_size)

def val_dataloader(self):
   val_dataset = torchvision.datasets.ImageFolder("./data/dataset_name/val", transform=self.transform)
   return torch.utils.data.DataLoader(val_dataset, batch_size=self.hparams.batch_size)
```
With `dataset_name` the name of the new dataset folder (saved in the data folder).

*Note*: During training, check that your model is training on the GPU and note the CPU, otherwise, this will take lot of time to train.

At the end of the training, a `weights.pt` file will be saved containing the trained network. It can be loaded with `ImageClassifier.load_from_checkpoint("weights.pt")`. To use it, modify the `get_model` function from `src/model.py` by adding the path of the new model:

```python
def get_model(path: str) -> ImageClassifier:
    """
    Load a saved pytorch lightning model from a path.
    The model is put in eval mode and gradient is deactivated.
    :param path: path where is saved the model
    :return: ImageClassifier model with loaded weitgh.
    """
    model = ImageClassifier.load_from_checkpoint(r"model/weights.pt")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    return model
```
