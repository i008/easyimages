import os
import sys

import PIL
import torchvision

sys.path.append('..')

import pytest
from PIL import Image
from torchvision.transforms import ToTensor, transforms

from easyimages import EasyImage, EasyImageList
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


def test_from_pytorch_batch():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    Trans = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    torch_image = Trans(PIL.Image.open('./tests/test_data/image_folder/img_00000003.jpg'))

    image3 = EasyImage.from_torch(torch_image, mean=MEAN, std=STD)

    assert isinstance(image3, EasyImage)


def test_from_glob():
    ilist = EasyImageList.from_glob('tests/test_data/image_folder/*.jpg')
    assert len(ilist) == 3


def test_html_rendering():
    ilist = EasyImageList.from_glob('tests/test_data/image_folder/*.jpg')

    html = ilist.visualize_grid_html(ilist.images)

    assert 'img_00000001' in html


def test_easyimage_from_pil():
    Im = PIL.Image.open(
        os.path.join(dir_path, './test_data/hierarchy_images/Boston_Celtics_Graphic_Tee/img_00000002.jpg'))
    ei = EasyImage.from_pil(Im)

    assert 'img_0000' in ei.name
    assert 'test_data/hierarchy_images/Boston_Celtics_Graphic_Tee/img_00000002.jpg' in str(ei.uri)


def test_easylist_from_pil():
    Im = PIL.Image.open(
        os.path.join(dir_path, './test_data/hierarchy_images/Boston_Celtics_Graphic_Tee/img_00000002.jpg'))

    List = EasyImageList.from_pil([Im] * 10)

    assert 'img_0000' in List[0].name
