import io
import os
import torch
from itertools import cycle
import PIL
import matplotlib as mpl
import torchvision

mpl.use('agg')

import numpy as np
from IPython import get_ipython
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

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


# def visualize_bboxes(image, boxes, threshold=0.1, return_format='PIL'):
#     if not isinstance(image, np.ndarray):
#         # othweriwse assume PIL
#         image = np.array(image)
#
#     cycol = cycle('bgrcmk')
#     detection_figure = plt.figure(frameon=False)
#     dpi = mpl.rcParams['figure.dpi']
#
#     imrows, imcols = image.shape[0], image.shape[1]
#     detection_figure.set_size_inches((imrows / dpi) * 1.5, (imcols / dpi) * 1.5)
#     current_axis = plt.Axes(detection_figure, [0., 0., 1., 1.])
#     current_axis.set_axis_off()
#     detection_figure.add_axes(current_axis)
#     current_axis.imshow(image)
#
#     for box in boxes:
#         if box.score and box.score < threshold:
#             continue
#
#         label = '{0} {1:.2f}'.format(box.label_name, box.score or '1')
#         color = next(cycol)
#         line = 4
#         current_axis.add_patch(
#             plt.Rectangle((box.x1, box.y1),
#                           box.x2 - box.x1,
#                           box.y2 - box.y1,
#                           color=color,
#                           fill=False, linewidth=line))
#
#         current_axis.text(box.x1, box.y1, label, size='x-large', color='white',
#                           bbox={'facecolor': color, 'alpha': 1.0})
#
#     current_axis.get_xaxis().set_visible(False)
#     current_axis.get_yaxis().set_visible(False)
#     plt.close()
#
#     if return_format == 'PIL':
#         return figure2img(detection_figure).convert('RGB')
#
#     elif return_format == 'NP':
#         return np.array(figure2img(detection_figure))[:, :, :3]
#
#     else:
#         return detection_figure


def xyxy2xywh(bb):
    x, y = bb[0], bb[1]
    width = bb[2] - bb[0] + 1
    height = bb[3] - bb[1] + 1
    return x, y, width, height


def change_box_order(boxes, input_order='tlbr', output_order='cwh', target_type=None):
    '''Change box order between:

    input_order:
    tlbr (top-left-bottom-right, x1y1x2y2)
    cwh  (center-width-height, (xywh)) center is in the middle of the box

    valid output order:
    all of the above plus:
    tlwh (top-left-width-height) this format is common in matplotlib

    '''
    if input_order == 'tlwh':
        raise NotImplementedError
    assert input_order != output_order
    assert input_order in ['tlbr', 'cwh']
    assert output_order in ['tlbr', 'cwh', 'tlwh']

    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
        cat = np.concatenate
        stack = np.stack
    elif isinstance(boxes, torch.Tensor):
        cat = torch.cat
        stack = torch.stack
    elif isinstance(boxes, np.ndarray):
        cat = np.concatenate
        stack = np.stack

    if input_order == 'cwh':
        a = boxes[:, :2]
        b = boxes[:, 2:]
        boxes = cat([a - b / 2, a + b / 2], 1)

    # transforms from xyxy/tlbr to any
    a = boxes[:, :2]
    b = boxes[:, 2:]

    if output_order == 'tlwh':
        return cat([a, b - a], axis=1)


    elif output_order == 'cwh':
        boxes = cat([(a + b) / 2, b - a + 1], 1)

    else:
        raise ValueError("")

    return boxes


def vis_boxes_on_image(img, boxes=None, label_names=None, scores=None, box_order='tlbr'):
    '''Visualize a color image.
    Args:
      img: (PIL.Image/tensor) image to visualize.
      boxes: (tensor) bounding boxes, sized [#obj, 4].
      label_names: (list) label names.
      scores: (list) confidence scores.
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_bbox.py
      https://github.com/chainer/chainercv/blob/master/chainercv/visualizations/vis_image.py
    '''
    # Plot image
    fig = plt.figure()
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
    # plt.axis('off')
    # plt.show()
    return fig


def get_pil_font(font_size):
    font = ImageFont.truetype(os.path.join(dir_path, "Verdana.ttf"), font_size)
    return font


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
