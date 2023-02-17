from train import ImageClassifier
import torchvision.transforms as T
import torch


def get_model(path: str) -> ImageClassifier:
    """
    Load a saved pytorch lightning model from a path.
    The model is put in eval mode and gradient is deactivated.
    :param path: path where is saved the model
    :return: ImageClassifier model with loaded weitgh.
    """
    model = ImageClassifier.load_from_checkpoint(path)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    return model


def get_transform() -> T.transforms.Compose:
    """
    Return the transform function to process the images for the model
    :return: transform function
    """
    transform =T.Compose([ T.Resize((224,224)),#T.Resize((196, 196)),
                           T.ToTensor(),
                           T.ConvertImageDtype(torch.float),
                           #T.Normalize(mean=(0.485, 0.456, 0.406),
                           #            std=(0.229, 0.224, 0.225))
                           ])
    #T.Compose([T.Resize((196, 196)),
    #                       T.ToTensor(),
    #                       T.ConvertImageDtype(torch.float),
    #T.Normalize(mean=(0.485, 0.456, 0.406),
    #            std=(0.229, 0.224, 0.225))
    return transform


def predict(model: ImageClassifier, images: torch.Tensor) -> torch.Tensor:
    """
    Apply the model to the images and return the predictions.
    :param model: Model used to predict.
    :param images: Inputs tensors
    :return:
    """
    prediction = model(images)
    softmax = torch.nn.Softmax(dim=1)
    prediction = softmax(prediction)
    return prediction
