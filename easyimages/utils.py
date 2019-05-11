import io
import os
import pathlib

import torch
import PIL
import matplotlib as mpl
import torchvision

mpl.use('agg')

import numpy as np
from IPython import get_ipython
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import requests


dir_path = os.path.dirname(os.path.realpath(__file__))


def pil_resize_not_destructive(pil_image, width):
    """

    Parameters
    ----------
    pil_image
    width

    Returns
    -------

    """
    wpercent = (width / float(pil_image.size[0]))
    hsize = int((float(pil_image.size[1]) * float(wpercent)))
    img = pil_image.resize((width, hsize), PIL.Image.ANTIALIAS)

    return img


def fig2img_v2(fig):
    """Converts matplotlib fig to PIL.Image
    Args:
        fig(`matplotlib.pyplot.figure`): Any matplotlib figure.
    Returns:
        `PIL.Image`: figure, converted to PIL Image.
    Examples:
        Create a figure:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import seaborn as sns
        >>> fig = plt.figure(figsize=(16,12))
        >>> sns.distplot(np.random.random(100))
        Convert to PIL.
        >>> pil_figure = fig2pil(fig)
    Note:
        On some machines, using this function has cause matplotlib errors.
        What helped every time was to change matplotlib backend by adding the following snippet
        towards the top of your script:
        >>> import matplotlib
        >>> matplotlib.use('Agg')
    """
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    w, h, _ = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def figure2img(f):
    """
    Converts a Matplotlib plot into a PNG image.

    Parameters
    ----------
    f: Matplotlib plot
        Plot to convert

    Returns
    -------
    Image
        PNG Image
    """

    buf = io.BytesIO()
    f.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    im = Image.open(buf)
    return im


def denormalize_img(image, mean, std):
    import torch

    return image * torch.Tensor(np.array(std).reshape(3, 1, 1)) + \
           torch.Tensor(np.array(mean).reshape(3, 1, 1))


def xyxy2xywh(bb):
    x, y = bb[0], bb[1]
    width = bb[2] - bb[0] + 1
    height = bb[3] - bb[1] + 1
    return x, y, width, height


def change_box_order(boxes, input_order='tlbr', output_order='cwh', target_type=None):
    '''Change box order between:
    input_order:
    tlbr (top-left/bottom-right, x1y1x2y2)
    cwh  (center/width-height, (xywh)) center is in the middle of the box
    valid output order:
    all of the above plus:
    tlwh (top-left-width-height) this format is common in matplotlib
    '''

    if input_order == output_order:
        return boxes

    assert input_order in ['tlbr', 'cwh', 'tlwh']
    assert output_order in ['tlbr', 'cwh', 'tlwh']

    if isinstance(boxes, np.ndarray):
        cat = np.concatenate
    elif isinstance(boxes, list):
        boxes = np.array(boxes)
        cat = np.concatenate
    elif isinstance(boxes, torch.Tensor):
        cat = torch.cat

    # change order from cwh -> tlbr
    if input_order == 'cwh':
        a = boxes[:, :2]
        b = boxes[:, 2:]
        boxes = cat([a - b / 2, a + b / 2], 1)

    # change order from tlwh -> tlbr
    if input_order == 'tlwh':
        xy = boxes[:, :2]
        wh = boxes[:, 2:]
        boxes = cat([xy, xy + wh], 1)

    # transforms from tlbr to anys
    x1y1 = boxes[:, :2]
    x2y2 = boxes[:, 2:]

    if output_order == 'tlwh':
        return cat([x1y1, x2y2 - x1y1], axis=1)
    elif output_order == 'cwh':
        boxes = cat([(x1y1 + x2y2) / 2, x2y2 - x1y1 + 1], 1)
    elif output_order == 'tlbr':
        return boxes
    else:
        raise ValueError("wtf")

    return boxes


def visualize_one_coco(coco_dataset, base_path, image_id=None):
    if isinstance(coco_dataset, str):
        coco = COCO(coco_dataset)
    elif isinstance(coco_dataset, COCO):
        coco = coco_dataset
    else:
        ValueError("Wrong input dataset should be a path to COCO json or coco.COCO instance")

    if image_id is None:
        image_id = np.random.choice(coco.getImgIds())

    image = coco.loadImgs(ids=[image_id])
    pil_image = PIL.Image.open(os.path.join(base_path, image[0]['file_name']))
    annotations_ids = coco.getAnnIds(imgIds=[image_id])
    annotations = coco.loadAnns(annotations_ids)
    boxes = [a['bbox'] for a in annotations]
    categories = [str(a['category_id']) + '-' + str(coco.cats[a['category_id']]['name']) for a in annotations]
    f = vis_image(pil_image, boxes, label_names=categories, box_order='tlwh')
    return f


def vis_image(img, boxes=None, label_names=None, scores=None, box_order='tlbr', axis_off=False, figsize=(15, 10)):
    """

    :param figsize:
    :param img: PIL.Image
    :param boxes: [[x1,y1,x2,y2], ... ]
    :param label_names:  ['car','dog' ... ]
    :param scores:  [0.5, 1]
    :param box_order: 'tlbr', 'tlwh'
    :param axis_off:
    :return:
    """

    # Plot image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, torch.Tensor):
        img = torchvision.transforms.ToPILImage()(img)
    ax.imshow(img)

    boxes = change_box_order(boxes, input_order=box_order, output_order='tlwh')

    # Plot boxes
    if boxes is not None:
        for i, bb in enumerate(boxes):
            x, y, w, h = bb

            ax.add_patch(plt.Rectangle(
                (x, y), w, h, fill=False, edgecolor='red', linewidth=2))

            caption = []
            if label_names is not None:
                caption.append(label_names[i])

            if scores is not None:
                caption.append('{:.2f}'.format(scores[i]))

            if len(caption) > 0:
                ax.text(bb[0] - 10, bb[1] - 10,
                        ': '.join(caption),
                        style='italic',
                        bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 2})

    # Show
    if axis_off:
        plt.axis('off')
    return fig


def get_pil_font(font_size):
    font = ImageFont.truetype(os.path.join(dir_path, "Verdana.ttf"), font_size)
    return font


def load_url_uri_to_pil(url_uri):
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


def make_notebook_wider():
    from IPython.display import HTML, display
    display(HTML("<style>.container { width:100% !important; }</style>"))


def get_execution_context():
    context = get_ipython().__class__.__name__
    if context == 'NoneType':
        return "regular_python"

    if context == 'InteractiveShellEmbed':
        return "terminal"

    if context == 'ZMQInteractiveShell':
        return "jupyter"

    print(type(context))


def draw_text_on_image(image, text, font_size):
    draw = ImageDraw.Draw(image)
    draw.text((50, 0), text, font=get_pil_font(font_size), fill=(0, 0, 0))
    return None
