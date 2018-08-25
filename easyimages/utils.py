
import io
from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


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
    return image * torch.Tensor(np.array(std).reshape(3, 1, 1)) + \
           torch.Tensor(np.array(mean).reshape(3, 1, 1))


def visualize_bboxes(image, boxes, threshold=0.1, return_format='PIL'):
    if not isinstance(image, np.ndarray):
        # othweriwse assume PIL
        image = np.array(image)

    cycol = cycle('bgrcmk')
    detection_figure = plt.figure(frameon=False)
    dpi = mpl.rcParams['figure.dpi']

    imrows, imcols = image.shape[0], image.shape[1]
    detection_figure.set_size_inches((imrows / dpi) * 1.5, (imcols / dpi) * 1.5)
    current_axis = plt.Axes(detection_figure, [0., 0., 1., 1.])
    current_axis.set_axis_off()
    detection_figure.add_axes(current_axis)
    current_axis.imshow(image)

    for box in boxes:
        if box.score and box.score < threshold:
            continue

        label = '{0} {1:.2f}'.format(box.label_name, box.score or '1')
        color = next(cycol)
        line = 4
        current_axis.add_patch(
            plt.Rectangle((box.x1, box.y1),
                          box.x2 - box.x1,
                          box.y2 - box.y1,
                          color=color,
                          fill=False, linewidth=line))

        current_axis.text(box.x1, box.y1, label, size='x-large', color='white',
                          bbox={'facecolor': color, 'alpha': 1.0})

    current_axis.get_xaxis().set_visible(False)
    current_axis.get_yaxis().set_visible(False)
    plt.close()

    if return_format == 'PIL':
        return figure2img(detection_figure).convert('RGB')

    elif return_format == 'NP':
        return np.array(figure2img(detection_figure))[:, :, :3]

    else:
        return detection_figure


def make_notebook_wider():
    from IPython.display import HTML, display
    display(HTML("<style>.container { width:100% !important; }</style>"))
