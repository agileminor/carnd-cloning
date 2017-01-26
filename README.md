# carnd-cloning
start.py -> file for creating initial building blocks, will rename to match project requirements later

Initial flow:
- set up a pipeline using smaller dataset loaded into memory, to find any bugs in the preprocessing, data loading, setting up model, etc.
- once initial pipeline is running, need to try out different models and different data augmentation. Tried out Nvidia and Comma.ai models, modifying normalization to match my pipeline. Both performed similarly, so decided to use Nvidia model for now and try data augmentation

Current flow:
- with data augmentation, I'm using more total images. To avoid having to have them in memory at the same time, I've switched to using generators

- preprocessing
	- image is trimmed to drop lower section (with car hood) and upper section (above road)
	- pixel values are normalized to +/- 0.5 (in a Lambda layer so that same normalization is done to training/simulation images)
	- training images are changed to RGB to match simulation images

- model modifications
	 - using Nvidia model from https://arxiv.org/pdf/1604.07316v1.pdf
	- modifications to model:
		- added dropout layers to help with overfitting
		- added a Lambda layer to handle data normalization
- currently added data augmentation
	- random horizontal image flip (50% odds of flip)
	- random brightness modification (0.3 - 1.3 of original image)
	- use left/right images + steering offset in addition to center image, with even changes of each image being used

- data augmentation to try:
	- small horizontal shifts with steering offsets
	- small vertical shifts, to help with slopes
	- random shadows, this is mainly for track 2

- other things to try:
	- shift all images to HSV space
	- differing amounts of steering offset for left/right images
	- excluding images with small steering angles
		- possibly train with one setting, then refine by adding or removing small angle images
	- vary amount of images per epoch
	- vary amounts of dropout used
	- vary number of epochs
	- changing image size. Currently using 200x66 (same as Nvidia model)
	- generate more raw data. Currently using Udacity dataset since I don't have a joystick to generate smoother driving. It should be possible to create a working model just using the Udacity data + augmentation

Experiments:
	- vary steering offset for left/right images from 0 to 0.5
	- results:
		- similar output checking, accuracy
		- best sim results from run 6/9 - 0.35, 0.5 offset, but only slightly better than previous 0.25 results
	- very threshold to include images with -0.1 < steering angle < 0.1
	- initial results with 0 threshold (all small angles excluded) has very low acc
