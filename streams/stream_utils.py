import os
import time
from threading import Thread
import re
import cv2
import numpy as np

LOCAL_INSTALLATION = True
RETRIEVE_EVERY_N_FRAMES = 5
USE_PAFY = False
CUSTOM_FPS = 30

if LOCAL_INSTALLATION:
    from libs.pytube import YouTube
else:
    from pytube import YouTube

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', custom_fps=None, use_oauth=False, allow_oauth_cache=False):
        self.mode = 'stream'
        self.terminated = False

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            url = eval(s) if s.isnumeric() else s
            if "youtube.com/" in str(url) or "youtu.be/" in str(url):  # if source is YouTube video
                if USE_PAFY:
                    check_requirements(("pafy", "youtube_dl"))
                    import pafy
                    url = pafy.new(url).getbest(preftype="mp4").url
                else:
                    yt = YouTube(url, use_oauth=use_oauth, allow_oauth_cache=allow_oauth_cache)
                    url = yt.streams.filter(file_extension="mp4", res=720).first().url

            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if custom_fps is not None:
                self.fps = custom_fps
            else:
                self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {self.fps:.2f} FPS).")
            thread.start()

        print("")  # newline

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            success = cap.grab()
            if success and n == RETRIEVE_EVERY_N_FRAMES:  # read every nth frame
                success, im = cap.retrieve()
                if success:
                    self.imgs[index] = im if success else self.imgs[index] * 0
                    n = 0

            time.sleep(1 / self.fps)  # wait time

            if not success:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Failed to retrieve frame from stream. Terminating...")
        self.terminated = True

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img = np.ascontiguousarray(img0).squeeze()
        return img

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
