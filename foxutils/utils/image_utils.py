from os import sep
from os.path import join as pathjoin
from . import core_utils

def read_image_to_tensor(filename, dataset_dir, im_height=None, im_width=None):
    from torchvision import transforms
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    image = Image.open(pathjoin(dataset_dir, filename))
    if im_width is not None and im_height is not None:
        image = image.resize((im_height, im_width))
    image = transforms.PILToTensor()(image)
    image = image.float()
    image /= 255.0
    return image

def write_image(full_filepath, project_path, target_image, target_folder=''):
    import cv2

    folder = pathjoin(project_path, target_folder)
    folder_path = pathjoin(folder, full_filepath.split(sep)[-1])
    core_utils.mkdir_if_not_exist(folder_path)
    cv2.imwrite(folder_path, target_image)