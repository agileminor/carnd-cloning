# carnd-cloning

model.py components:

run_nn - main function. Creates a model, trains it and saves the output json/h5

train_nn_gen: trains an nn model using generator driven data
    model - nn model to train
    batch_size - # of images to use per batch
    n_epoch - # of epochs to run training for
    img_shape - shape of images to train on
    offset: (optional) - value to +/- for left/right camera images. default is 0.25
    lr: (optional) - learning rate to pass to optimizer. default is 0.001

train_nn : trains an nn model with preloaded data

create_nn: creates basic nn model, used for pipecleaning only

create_nn_comma: creates a nn model based on the Comma.AI model

create_nn_nvidia: creates a nn model based on Nvidia model, with added Lambda for normalization and dropout layers added

get_batch_data: generator to return # of images for training model
    - folder: folder containing data
    - df: DataFrame with location of images, steering angles
    - num: number of images/labels to return for batch
    - img_shape: shape of images to return
    - augment: (optional) - boolean to decide to do data augmentation. Default is False
    - threshold: value to be used to determine to include small steering angle images or not. default is 1.0 (100% included)
    - offset: (optional) - value to +/- for left/right camera images. Default is 0.25

pick_image: returns left, right or center image with equal distribution

add_bright: modify brightness of image by 0.3 - 1.3

get_test_img: loads a few test images to evaluate current steering predictions

shift_image: shifts image in both x and y for data augmentation

explore_nn: train multiple models, changing one parameter and saving results to explore the solution space


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
	- small horizontal shifts with steering offsets
	- small vertical shifts, to help with slopes

- data augmentation to try:
	- random shadows, this is mainly for track 2

- other things to try:
	- excluding images with small steering angles
		- possibly train with one setting, then refine by adding or removing small angle images
	- vary amounts of dropout used
	- vary number of epochs
	- generate more raw data. Currently using Udacity dataset since I don't have a joystick to generate smoother driving. It should be possible to create a working model just using the Udacity data + augmentation

Experiments:
- vary steering offset for left/right images from 0 to 0.5
	- results:
		- similar output checking, accuracy
		- best sim results from run 6/9 - 0.35, 0.5 offset, but only slightly better than previous 0.25 results
- vary threshold to include images with -0.1 < steering angle < 0.1
	- initial results with 0 threshold (all small angles excluded) has very low acc, but predictions on 3 test images are better than higher acc runs!
	- no significant difference between the runs, except threshold of 0.6 performed worse than the rest

- vary samples per epoch from 4096 to 45056
	- most models did very poorly, quick turned off the road. Best were 4096, 16384, 32768 samples per epoch. Best overall was 32768, but none were as good as previous best 

- try 64x64 image (instead of 200x66, default Nvidia size)
	- results no different

- try using HSV instead of RGB
	- no improvement

- tried adding horizontally shifted images
	- model performed significantly worse, both in terms of reported acc/loss and simulation

- tried various epochs, all with samples per epoch of 32K
	- no improvement with different # of epochs, above 8 epochs all output values are the same
