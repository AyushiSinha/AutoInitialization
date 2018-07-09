## Automatic initialization of endoscope in canonical coordinate frame given endoscopic video frame
#### This software allows a user to generate data simulating a moving endoscope in a textured nasal cavity (or other anatomy) mesh and to train a neural network to learn the regions in which the endoscope must be located to generate particular images.


**Dependencies**:
- [OpenGL](https://pypi.org/project/PyOpenGL/): pip install PyOpenGL PyOpenGL_accelerate
- [TriMesh](https://trimsh.org/): pip install trimesh
- [PIL](http://www.pythonware.com/products/pil/): pip install Pillow
- [Torch](https://pypi.org/project/torchvision/): pip install torchvision
- [Scikit-image](http://scikit-image.org/): pip install scikit-image


**Run**:
- Clone this repository
- Run *python viewer.py* to render default mesh
    * Mesh and texture image can be modified in *opengl_viewer/opengl_viewer.py*
    * Move the camera inside the mesh using WQAZSX and arrow keys
    * Key c saves the current view as an image and the current camera pose in a text file, each pair known as a keyframe
- *run_data_collection.sh* calls *collect_data.py* to interpolate between saved keyframes and save renderings at different camera poses
- *view_training_data.py* displays the saved images in a given folder allowing the user to inpect the training images
- *scene_classifier.py* trains a neural network to learn the region that a camera should lie in to generate different images
