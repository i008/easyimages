
# easyimages

![Build Test Release](https://github.com/i008/easyimages/workflows/Build%20Test%20Release/badge.svg)

# Info

This small but handy package solves several issues i had while working with images and image datasets - especially in the context
of exploring datsets, inspecting and shareing the results.
Keep in mind that his package is not directly related to the training process and loading
image data, for that i found pytorch dataloading patterns to work very well.

# Installation
```bash
pip install easyimages
```


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




Credits
-------
MAP calculation code comes from: 
https://github.com/MathGaron/mean_average_precision
