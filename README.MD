
# easyimages

[![Foo](https://img.shields.io/pypi/v/easyimages.svg)](https://pypi.python.org/pypi/easyimages)
[![Foo](https://img.shields.io/travis/i008/easyimages.svg)](https://travis-ci.org/i008/easyimages)
[![Foo](https://pyup.io/repos/github/i008/easyimages/shield.svg)](https://pyup.io/repos/github/i008/easyimages/)


# Images made easy

This small but handy package solves several issues i had while working with images and image datasets - especially in the context
of exploring datsets,training neural networks and exploring results.
Keep in mind that  his package is not directly related to the training process and loading
image data, for that i found pytorch dataloading patterns work very well.


Features
--------
- Simple API
- Easy image exploration
- Inteligent behaviour based on execution context (terminal, jupyter etc)
- Lazy evaluation
- Loading images from many different sources (filesystem, pytorch, numpy, web-urls, etc)
- Storing annotations (tags, bounding boxes) allong the image in the same object
- Visualizing labels (drawing boxes and drawing the label onto the image)
- Visualizing images as Grids (ImagesLists)
- Visualizing huge amounts of images at once (by leveraging fast html rendering)
- Displaying images while working in jupyter notebook
- Displaying images inline in console mode (iterm)



Examples
--------

For detailed examples check the examples notebook



