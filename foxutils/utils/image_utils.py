from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

from os.path import join as pathjoin

def read_image_to_tensor(filename, dataset_dir, im_height=None, im_width=None):
    image = Image.open(pathjoin(dataset_dir, filename))
    if im_width is not None and im_height is not None:
        image = image.resize((im_height, im_width))
    image = transforms.PILToTensor()(image)
    image = image.float()
    image /= 255.0
    return image
