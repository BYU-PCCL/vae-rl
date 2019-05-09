# Unsupervised Creation of Robust Environment Representations for Reinforcement Learning

Rainbow contains the code for the reinforcement learning algorithm. I addded the following arguments to it:
1. use-encoder: whether or not to use the VLAE. Option 0 means that the VLAE is not used. Option 1 means that Rainbow is only trained on the latent space extracted from the VLAE. Option 2 means that Rainbow trains on the latent space concatenated to the original image.
2. output-name: a string to concatenate onto the end of the results name to distinguish experiments.
3. use-convcoord: whether or not to add a ConvCoord layer at the beginning of the model.
4. name: name of the VLAE model to load.
5. encode-transitions: compress 4 black and white images at a time.

The VLAE folder contains the model used to train the VLAE. Once it is trained, I copy over the weights to Rainbow/VLAE.

The paper folder contains all the images and files to generate a report of this project.

Loss plots, Q & result plots, saved models, and saved profiling information of the 4 transitions experiment on Seaquest can be found in mnt/pccfs/backed_up/mckell/vae-rl.

Code for the frozen experiments can be found in mnt/pccfs/backed_up/mckell/Rainbow.

Old code (from the time of Maria and Sydney) can be found in mnt/pccfs/backed_up/mckell/vlae_old.

The dataset can be found in mnt/pccfs/not_backed_up/atarigames. 
every_timestep contains raw observations. 
grayscale contains the grayscale version of the raw observations. 
transitions contains 4 consecutive graysclae images concatentated together. 
delete_black.py deletes all observations that are all black. 
grayscale.py turns images into grayscale ones. 
make_transitions.py turns concatenates consecutive images.
screenshot_runs.py retreives the observations from simulated episodes.

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
