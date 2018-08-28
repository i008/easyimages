import os
import sys
sys.path.append('..')

import pytest
from PIL import Image
from torchvision.transforms import ToTensor

from easyimages import EasyImage
from easyimages import bbox


dir_path = os.path.dirname(os.path.realpath(__file__))


def test_easy_image_from_fie():
    p = os.path.join(dir_path, './test_data/hierarchy_images/Boston_Celtics_Graphic_Tee/img_00000002.jpg')
    image = EasyImage.from_file(p, lazy=False)
    assert isinstance(image.image, Image.Image)


def test_easy_image_from_url():
    url = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"

    image = EasyImage.from_url(url)

    assert isinstance(image.image, Image.Image)


def test_lazy_image_from_url():
    url = "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"

    image = EasyImage.from_url(url, lazy=True)

    assert isinstance(image.image, type(None))

    image.download()

    assert isinstance(image.image, Image.Image)


def test_easy_image_from_torch():
    image = Image.open(os.path.join(dir_path, './test_data/image_folder/img_00000001.jpg'))

    torch_image = ToTensor()(image)

    easy_image = EasyImage.from_torch(torch_image)

    assert isinstance(easy_image.image, Image.Image)


def test_error_when_trying_to_draw_boxes_when_not_provided():
    boxes = [bbox(10, 10, 75, 75, 1, 'class1'), bbox(20, 20, 95, 95, 1, 'class2')]

    p = os.path.join(dir_path, './test_data/hierarchy_images/Boston_Celtics_Graphic_Tee/img_00000002.jpg')
    easy_image_old = EasyImage.from_file(p, lazy=False)
    with pytest.raises(AssertionError):
        easy_image_old.draw_boxes()

    easy_image_new = EasyImage.from_file(p, lazy=False, boxes=boxes)

    assert easy_image_new != easy_image_old
