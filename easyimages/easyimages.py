# -*- coding: utf-8 -*-

import io
import os
import pathlib
import urllib
import uuid
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import PIL
import numpy as np
import requests
import torchvision
from IPython.display import HTML, display
from PIL import Image
from imutils.convenience import build_montages
from easyimages.utils import denormalize_img, visualize_bboxes, get_execution_context, draw_text_on_image
import matplotlib.pyplot as plt

import subprocess


bbox = namedtuple('bbox_abs', ['x1', 'y1', 'x2', 'y2', 'score', 'label_name'])
label = namedtuple('label', ['label'])
CTX = get_execution_context()


dir_path = os.path.dirname(os.path.realpath(__file__))



class EasyImage:
    def __repr__(self):
        return str('EasyImageObject: {} | labels: {} | downloaded: {} | size: {} |'.format(
            self.name or "unknown",
            self.label,
            self.downloaded,
            self.size))

    def __init__(self, image: PIL.Image,
                 name=None,
                 url=None,
                 uri=None,
                 boxes=None,
                 label=None,
                 mask=None):

        assert isinstance(boxes, (list, type(None)))
        assert isinstance(label, (list, type(None)))

        if name:
            self.name = name
        if url:
            assert 'http' in url
            self.name = self._name_from_url(url)
        if uri:
            assert isinstance(uri, pathlib.Path)
            self.name = uri.name
        if image is None and (not url and not uri):
            raise ValueError("Image is in a not downloaded state url or uri is required")
        if image is None and (url or uri):
            self.downloaded = False
        if image is not None:
            self.downloaded = True

        self.url = url
        self.uri = uri
        self.image = image
        self.boxes = boxes or []
        self.label = label or []
        self.download_error = False
        self.mask = mask

    def show(self, inline=False):
        self.download()
        print(self.__repr__())
        if (CTX == 'terminal' or CTX == 'python') and not inline:
            self.image.show()
        if (CTX == 'terminal' or CTX == 'python') and inline:
            self.show_inline()
        if CTX == 'regular_python':
            self.image.show()
        else:
            return self.image

    def show_inline(self):
        temp_file = os.path.join(dir_path, 'tmp.png')
        self.save(dir_path, 'tmp.png')
        subprocess.call(['bash', os.path.join(dir_path, 'imgcat.txt'), temp_file])
        os.remove(temp_file)

    @property
    def size(self):
        if self.image:
            return self.image.size

    def add_boxes(self, box_es):
        pass

    @staticmethod
    def _name_from_url(url):
        name = os.path.basename(urllib.parse.urlparse(url).path)
        return name

    @staticmethod
    def _load_url_uri_to_pil(url_uri):
        """

        Parameters
        ----------
        url_uri

        Returns
        -------
        PIL.Image

        """
        if isinstance(url_uri, pathlib.Path):
            url_uri = str(url_uri)

        if url_uri.startswith('http'):
            response = requests.get(url_uri)
            image_original = io.BytesIO(response.content)
            image_pil = Image.open(image_original).convert('RGB')
        else:
            image_pil = Image.open(url_uri).convert('RGB')
        return image_pil

    @classmethod
    def from_file(cls, file_path: [str, pathlib.Path], lazy=False, *args, **kwargs):
        image = None
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not lazy:
            image = cls._load_url_uri_to_pil(str(file_path))
        return cls(image, uri=file_path, *args, **kwargs)

    @classmethod
    def from_url(cls, file_url, lazy=False, *args, **kwargs):
        name = kwargs.get('name')
        image = None
        if not name:
            name = os.path.basename(
                urllib.parse.urlparse(file_url).path)
        if not lazy:
            image = cls._load_url_uri_to_pil(file_url)
        return cls(image, name=name, url=file_url, *args, **kwargs)

    @classmethod
    def from_torch(cls, tensor, name=None, mean=None, std=None, *args, **kwargs):
        if mean and std:
            tensor = denormalize_img(tensor, mean=mean, std=std)
        image = torchvision.transforms.ToPILImage()(tensor)
        if name is None:
            name = str(uuid.uuid4())[:8] + '.jpg'
        return cls(image, name=name, *args, **kwargs)

    def from_tensorflow(cls, tensor, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_numpy(cls, array, *args, **kwargs):
        image = PIL.Image.fromarray(array)
        return cls(image)

    def download(self, thread=True):
        if not self.downloaded:
            try:
                self.image = self._load_url_uri_to_pil(self.url or self.uri)
                self.downloaded = True
            except Exception as e:
                print(e)
                self.download_failed = True
        return self

    def draw_boxes(self, threshold=0.1):
        assert self.boxes, "Cant draw boxes if they are not provided"
        assert self.image
        self.image = visualize_bboxes(self.image, self.boxes, threshold=threshold)
        return self

    def show_boxes(self, threshold=0.1):
        assert self.boxes, "Cant draw boxes if they are not provided"
        assert self.image

        return visualize_bboxes(self.image, self.boxes, threshold=threshold)

    def draw_label(self, font_size=40):
        draw_text_on_image(self.image, str(self.label), font_size=font_size)
        return self

    def save(self, base_path: [str, pathlib.Path], name: [None, str] = None):
        if self.image is None:
            self.download()
        if isinstance(base_path, str):
            base_path = pathlib.Path(base_path)
        if name is None:
            name = self.name
        save_path = base_path / name
        self.image.save(save_path)
        self.uri = save_path

        return self


class EasyImageList:
    IMAGE_FILE_TYPES = ('*.jpg', '*.png', '*.tiff', '*.jpeg')
    GRID_TEMPLATE = "<img style='width: 100px; height: 100px; margin: 1px; float: left; border: 0px solid black;'title={label} src='{url}'/>"
    open_browser = CTX == 'terminal'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        return self.images[ix]

    def __iter__(self):
        return iter(self.images)

    def __init__(self, images, *args, **kwargs):
        self.images = images

        self.all_labels = list(set(list(chain(*[im.label for im in images]))))

    def __repr__(self):
        return "<ImageList with {} EasyImages>".format(len(self.images))

    def download(self):
        def _download(im):
            try:
                im.download()
            except:
                print("failed downloading")

        with ThreadPoolExecutor(100) as tpe:
            futures = tpe.map(_download, self.images)

        return self

    def draw_boxes(self):
        def _draw(im):
            try:
                im.draw_boxes()
            except:
                print("Failed drawing boxes")

        with ThreadPoolExecutor(100) as tpe:
            futures = tpe.map(_draw, self.images)
        return self

    @classmethod
    def from_folder(cls, path, download=True, *args, **kwargs):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        image_files = []
        for pattern in cls.IMAGE_FILE_TYPES:
            image_files.extend(path.glob(pattern))
        images = [EasyImage.from_file(image_path, download=download) for image_path in image_files]
        return cls(images, *args, **kwargs)

    @classmethod
    def from_multilevel_folder(cls, path, lazy=False):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        image_files = []
        for pattern in cls.IMAGE_FILE_TYPES:
            image_files.extend(path.glob('**/{}'.format(pattern)))

        images = []
        for image in image_files:
            rel_path = image.relative_to(path)
            class_name = rel_path.parents[0]
            images.append(EasyImage.from_file(image,
                                              label=[str(class_name)],
                                              lazy=lazy))
        return cls(images)

    @classmethod
    def from_list_of_images(cls, list_of_images):
        return cls(list_of_images)

    @classmethod
    def from_list_of_urls(cls, list_of_image_urls, download=True):
        ims = [EasyImage.from_url(url, download=download) for url in list_of_image_urls]
        return cls(ims)

    @classmethod
    def from_torch_batch(cls, batch):
        pass


    def visualize_grid_html(self,images, open_browser=open_browser):
        templates = []
        for image in images:
            p = image.uri or image.url
            if not 'http' in str(p) and open_browser:
                p = p.absolute()
            templates.append(self.GRID_TEMPLATE.format(url=p, label=image.label))
        html = ''.join(templates)
        if open_browser:
            import webbrowser
            p = os.path.join(os.path.expanduser('~'), 'vistmp.html')
            with open(p, 'w') as f:
                f.write(html)
            webbrowser.open('file://' + p)
        else:
            display(HTML(html))

    def to_html(self, by_class=True, custom_filter=None):
        if self.all_labels and by_class:
            for label_name in self.all_labels:
                print("Drawing {}".format(label_name))
                images = list(filter(lambda x: label_name in x.label, self.images))
                self.visualize_grid_html(images)
        else:
            self.visualize_grid_html(self.images)

    def visualize_one_by_one(self):
        import cv2
        for image in self.images:
            cv2.imshow('image', cv2.cvtColor(np.array(image.image),cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def visualize_grid_numpy(self, image_shape=100, montage_shape=None, inline=False):

        images_np = [np.array(im.download().image) for im in self.images]

        if montage_shape is None:
            n_cols = 6
            montage_shape = (n_cols, int(np.ceil(len(images_np) / n_cols)))

        montage = build_montages(images_np, image_shape=(image_shape, image_shape), montage_shape=montage_shape)

        easy_montage = EasyImage.from_numpy(montage[0])

        if CTX == 'terminal':
            easy_montage.image.show()

        if CTX == 'terminal' and inline:
            if 'xterm' not in os.environ.get('TERM', 'None'):
                raise ValueError("Inline mode only works in Iterm like emulators")
            easy_montage.show_inline()
        if CTX == 'jupyter':
            plt.figure(figsize=(15,15))
            plt.imshow(montage[0])

    def from_metadata_file(cls, metdata_file):
        pass
