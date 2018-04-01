"""
===========================
Structural similarity index
===========================

When comparing images, the mean squared error (MSE)--while simple to
implement--is not highly indicative of perceived similarity.  Structural
similarity aims to address this shortcoming by taking texture into account
[1]_, [2]_.

The example shows two modifications of the input image, each with the same MSE,
but with very different mean structural similarity indices.

.. [1] Zhou Wang; Bovik, A.C.; ,"Mean squared error: Love it or leave it? A new
       look at Signal Fidelity Measures," Signal Processing Magazine, IEEE,
       vol. 26, no. 1, pp. 98-117, Jan. 2009.

.. [2] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality
       assessment: From error visibility to structural similarity," IEEE
       Transactions on Image Processing, vol. 13, no. 4, pp. 600-612,
       Apr. 2004.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from PIL import Image

gt = np.asarray(io.imread('Test/Set5/rawimage-000259.bmp'))
gt = Image.open('Test/Set5/rawimage-000259.bmp').convert('L')
gt = gt.crop((1, 1, 484, 484))
img = img_as_float(gt)
rows = img.shape[0]
cols = img.shape[1]



def psnr(x, y):
    return np.linalg.norm(x - y)
srcnn = np.asarray(io.imread('Test/Set5/test_image.png'))
srcnn = Image.open('Test/Set5/test_image.png').convert('L')
srcnn = srcnn.crop((1, 1, 484, 484))
img_srcnn = img_as_float(srcnn)
casrcnn = np.asarray(io.imread('Test/Set5/test_image.png'))
casrcnn = Image.open('Test/Set5/test_image.png').convert('L')
casrcnn = casrcnn.crop((1, 1, 484, 484))
img_casrcnn = img_as_float(casrcnn)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                         sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

psnr_srcnn = psnr(img, img_srcnn)
ssim_srcnn = ssim(img, img_srcnn, data_range=img_srcnn.max() - img_srcnn.min())

psnr_casrcnn = psnr(img, img_casrcnn)
ssim_casrcnn = ssim(img, img_casrcnn,
                  data_range=img_casrcnn.max() - img_casrcnn.min())

label = 'PSNR: {:.2f}, SSIM: {:.2f}'

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)

ax[0].set_title('Original')

ax[1].imshow(img_srcnn, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(label.format(psnr_srcnn, ssim_srcnn))
ax[1].set_title('9-1-5 SRCNN')

ax[2].imshow(img_casrcnn, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(label.format(psnr_casrcnn, ssim_casrcnn))
ax[2].set_title('9-5-1-5 CASRCNN')

plt.tight_layout()
plt.show()
