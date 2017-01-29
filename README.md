# carnd-cloning

##model.py components:

explore_nn: train multiple models, changing one parameter and saving results to explore the solution space
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



##Initial flow:
- set up a pipeline using smaller dataset loaded into memory, to find any bugs in the preprocessing, data loading, setting up model, etc.
- once initial pipeline is running, need to try out different models and different data augmentation. Tried out Nvidia and Comma.ai models, modifying normalization to match my pipeline. Both performed similarly, so decided to use Nvidia model for now and try data augmentation

##Current flow:
- with data augmentation, I'm using more total images. To avoid having to have them in memory at the same time, I've switched to using generators

- best model so far - 64x64 image shape, HSV colour, 48 samples per epoch, 5+ epochs, using later epochs for tuning
- preprocessing
	- image is trimmed to drop lower section (with car hood) and upper section (above road)
	- pixel values are normalized to +/- 0.5 (in a Lambda layer so that same normalization is done to training/simulation images)
	- training and simulator images are changed to HSV, due to better results with HSV images

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

- data augmentation not tried:
	- random shadows, this is mainly for track 2


##Model detailed summary:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 64, 64, 3)     0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 30, 30, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 30, 30, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 13, 13, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 13, 13, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 5, 48)      43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 5, 48)      0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 3, 64)      27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 3, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 1, 64)      36928       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 1, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 1, 64)      0           elu_5[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          75660       flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1164)          0           elu_6[0][0]                      
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           elu_7[0][0]                      
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dropout_3[0][0]                  
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         elu_8[0][0]                      
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          elu_9[0][0]                      
====================================================================================================
Total params: 329,079
Trainable params: 329,079
Non-trainable params: 0

##Experiments:
- baseline is 5 epochs, 32K samples per epoch, 50 max shift, 0.25 offset, 1.0 threshold
- vary steering offset for left/right images from 0 to 0.5 
	- results:
		- similar output checking, accuracy
- vary threshold to include images with -0.1 < steering angle < 0.1 

- vary samples per epoch from 4096 to 45056 (needs to be repeated after reverse bug)
	- made first turn after 28K samples, good results at 48K

-vary number of epochs:
	- made first turn after 3 epochs, 2nd turn at 4 epochs, missed 2nd at 5, epoch 20 made it around right turn, all the way around once and then went off after bridge on 2nd pass - 2nd run of same model, did not make it past turn after bridge!
	- seems to need 4+ epochs. Using later epochs with lower learning rate to tune model

- try 64x64 image (instead of 200x66, default Nvidia size) 
	- results no different

- try using HSV instead of RGB 
	- no improvement

- try using HSV with 64x64 images
	- big improvement, now repeating other experiments with that baseline

- tried training with no small angle images, then refining model adding small angle images with either lower or same learning rate
	- initial results with 0 threshold (all small angles excluded) has very low acc, but predictions on 3 test images are better than higher acc runs!
	- runs with medium threshold (0.4-0.8) performed better, faster. Using value of 0.8 with tuning using 1.0

- tried training with different amounts of max shift for horizontal translation
	- only at 50 did I start to make the first turn
	- seems better at higher values - try 80-100 as default
