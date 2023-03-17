from torchvision import transforms, models
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transforming = transforms.Compose((transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])))
    
    img = Image.open(image)

    return transforming(img)
    

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path).unsqueeze(dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    # move input tensor to the same device as the model's weights
    image = image.to(device)

    with torch.no_grad():
      model.eval()
      output = model.forward(image)

    ps = torch.exp(output)
    # label = labels.cpu()
    ps, classes = ps.topk(topk, dim=1)
    # equals = classes == label.view(*top_class.shape)

    return ps, classes
