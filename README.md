# Unsupervised Creation of Robust Environment Representations for Reinforcement Learning

Rainbow contains the code for the reinforcement learning algorithm. I addded the following arguments to it:
1. use-encoder: whether or not to use the VLAE. Option 0 means that the VLAE is not used. Option 1 means that Rainbow is only trained on the latent space extracted from the VLAE. Option 2 means that Rainbow trains on the latent space concatenated to the original image.
2. output-name: a string to concatenate onto the end of the results name to distinguish experiments.
3. use-convcord: whether or not to add a ConvCoord layer at the beginning of the model.

The VLAE folder contains the model used to train the VLAE. Once it is trained, I copy over the weights to Rainbow/VLAE.

Packages need to run Rainbow:
* cv2 package - 
  * pip install opencv-python
  * apt-get update
  * apt-get install libgtk2.0-dev
* skimage -
	* pip install scikit-image
* matplotlib
* plotly
* atari_py
* tensorflow
* tqdm

If running on SIVRI
* pip install --upgrade torch torchvision

Packages needed to run VLAE:
* matplotlib
* tensorflow
* skimage
  * pip install scikit-image
