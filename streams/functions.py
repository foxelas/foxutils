from . import stream_utils
import cv2
from os.path import join as pathjoin
from ..utils import core_utils

UPDATE_RESULTS_EVERY_N_FRAMES = 10
DEFAULT_SAVE_DIR = pathjoin(core_utils.datasets_dir, 'youtube')


def save_frames(url_info, savedir, save_every_n_frames=UPDATE_RESULTS_EVERY_N_FRAMES):
    dataset = stream_utils.LoadStreams(url_info["url"], custom_fps=None)
    target_label = url_info["label"]
    target_name = url_info["name"]
    count = 0
    c = 0
    for image in dataset:
        count = count + 1
        c = c + 1
        if image is not None:
            if count == save_every_n_frames:
                count = 0
                filename = target_name + '_frame' + str(c) + '.jpg'
                target_path = pathjoin(savedir, target_label, '')
                core_utils.mkdir_if_not_exist(target_path)
                # cv2.imshow('image',image)
                # cv2.waitKey(0)
                fullpath = pathjoin(target_path, filename)
                if not cv2.imwrite(fullpath, image):
                    raise Exception("Could not write image")
                # print(f"Saved in {fullpath}")

        if dataset.terminated:
            break
    print(f'Finished processing stream.')


def bulk_download_from_youtube(url_list, savedir=None, save_every_n_frames=UPDATE_RESULTS_EVERY_N_FRAMES):
    """
    Example Url list :

    url_list = [
        {'name': 'Feb_2022', 'url': 'https://www.youtube.com/watch?v=t2QCCKYpqW0', 'label': 'storm'},
        {'name': 'Aug_2023', 'url': 'https://www.youtube.com/watch?v=SiTYtS9kxwY', 'label': 'sunny'},
        {'name': '2018', 'url': 'https://www.youtube.com/watch?v=8mqSJLPvLWg', 'label': 'rain'},
        {'name': '2020', 'url': 'https://www.youtube.com/watch?v=gMNAywukAto', 'label': 'rain'},
    ]
    """
    if savedir is None:
        savedir = DEFAULT_SAVE_DIR
    for x in url_list:
        save_frames(x, savedir, save_every_n_frames)
