from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os

root_dir = "training_dataset"
img_dir = os.path.join(root_dir, 'frame')
imgs = os.listdir(img_dir)
for i in range(0, len(imgs), 120):
    fig = plt.figure()
    for j in range(i, i+120):
#    for j in range(len(imgs)-120, len(imgs)):
	img_name = os.path.join(img_dir, imgs[j])
	image = io.imread(img_name)
	plt.subplot(10, 12, j%120+1)
	plt.imshow(image)
	plt.title(imgs[j])
    plt.show()
#    break


