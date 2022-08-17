import torch
import numpy as np
import pandas as pd


def generate_score(prediction: torch.Tensor, image_size: tuple, crop_size: tuple, img_pos: list) -> np.array:
    """
    Generate a table of prediction scores, this table can be superposed to the original images.
    :param prediction: model prediction
    :param image_size: tuple of the original dimension
    :param crop_size: tuple of the size of the sub images
    :param img_pos: list of the position of all sub images
    :return:
    """
    pred = torch.zeros((image_size[0] // crop_size[0], image_size[1] // crop_size[1], 5))

    for pos, pred_ in zip(img_pos, prediction):
        pred[pos] = pred_

    pred = torch.moveaxis(pred, 2, 0)

    score = pred.cpu().detach().numpy()
    score = score.round(decimals=2)
    return score


def get_classification(score: np.array) -> np.array:
    """
    Transform the score into a class.
    :param score:
    :return:
    """
    return np.argmax(score, axis=0)


def classification_name(best_id: int, table: list) -> str:
    """
  Extract the classification name a classificationn table.
  Ex:
  >>> classification_name(0 , ["class A", "class B"])
  >>> class A
  """
    best_id = int(best_id)
    return table[best_id]


def pandas_results(classification: np.array, score: np.array, class_names: list) -> pd.DataFrame:
    """
    Transform the results to a pandas dataframe usable by plotly for display
    :param classification: array of classification
    :param score: array of score
    :param class_names: names of the class
    :return:
    """
    dim_representation = classification.shape

    x, y, class_, bulging, edges, good, porous, powder = (list() for i in range(8))
    for i in reversed(range(dim_representation[0])):
        for j in range(dim_representation[1]):
            x.append(j)
            y.append(i)
            class_.append(str(classification[i][j]))
            bulging.append(str(score[0][i][j]))
            edges.append(str(score[1][i][j]))
            good.append(str(score[2][i][j]))
            porous.append(str(score[3][i][j]))
            powder.append(str(score[4][i][j]))

    df = pd.DataFrame(np.array([x, y, class_, bulging, edges, good, porous, powder]).T,
                      columns=["x", "y", "classification", "bulging", "corner", "good", "porous", "powder"])
    # Convert classification into readable names
    df['classification_name'] = df['classification'].map(lambda x: classification_name(x, class_names))

    # CLean the data for display
    df[["x", "y", "classification", "bulging", "corner", "good", "porous", "powder"]] = df[
        ["x", "y", "classification", "bulging", "corner", "good", "porous", "powder"]].apply(pd.to_numeric)
    df["x"] = df["x"] + 0.5
    df["y"] = df["y"] + 0.5
    return df
