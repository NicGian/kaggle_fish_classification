# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# If you want to use Theano, all you need to change
# is the dim ordering whenever you are dealing with
# the image array. Instead of
# (samples, rows, cols, channels) it should be
# (samples, channels, rows, cols)

# Keras stuff
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

# A large amount of the data loading code is based on najeebkhan's kernel
# Check it out at https://www.kaggle.com/najeebkhan/leaf-classification/neural-network-through-keras
root = '/dfs/home-scratch/nig/cnn/input'
np.random.seed(12)
split_random_state = 7
split = .85
max_img_dim = 224

def load_numeric_training(standardize=True):
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    # Read data from the CSV file
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')

    # Since the labels are textual, so we encode them categorically
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    # standardize the data by setting the mean to 0 and std to 1
    #X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return ID, y


def load_numeric_test(standardize=True):
    """
    Loads the pre-extracted features for the test data
    and returns a tuple of the image ids, the data
    """
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = test.pop('id')
    # standardize the data by setting the mean to 0 and std to 1
    #test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID


def resize_img(img, max_dim=max_img_dim):
    """
    Resize the image to so the maximum side is of size max_dim
    Returns a new image of the right size
    """
    # Get the axis with the larger dimension
    max_ax = max((0, 1), key=lambda i: img.size[i])
    # Scale both axes so the image's largest dimension is max_dim
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=max_img_dim, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the output array
    # NOTE: Theano users comment line below and
    X = np.empty((len(ids), max_dim, max_dim, 3))
    # X = np.empty((len(ids), 1, max_dim, max_dim)) # uncomment this
    for i, idee in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(os.path.join(root, 'images', str(idee)), grayscale=False), max_dim=max_dim) #+ '.jpg'
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        # NOTE: Theano users comment the two lines below and
        length = x.shape[0]
        width = x.shape[1]
        # length = x.shape[1] # uncomment this
        # width = x.shape[2] # uncomment this
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        # Insert into image matrix
        # NOTE: Theano users comment line below and
        X[i, h1:h2, w1:w2, 0:3] = x
        # X[i, 0:1, h1:h2, w1:w2] = x  # uncomment this
    # Scale the array values so they are between 0 and 1
    #return np.around(X / 255.0)
	return np.around(X - 128.0) #VGG16 preprocessing is only mean subtraction


def load_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted feature and image training data and
    splits them into training and cross-validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # Load the pre-extracted features
    ID, y = load_numeric_training()
    # Load the image data
    X_img_tr = load_image_data(ID)
    # Split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(y,y))
    X_img_val, y_val = X_img_tr[test_ind], y[test_ind]
    X_img_tr, y_tr = X_img_tr[train_ind], y[train_ind]
    return (X_img_tr, y_tr), (X_img_val, y_val)


def load_test_data():
    """
    Loads the pre-extracted feature and image test data.
    Returns a tuple in the order ids, pre-extracted features,
    and images.
    """
    # Load the pre-extracted features
    ID = load_numeric_test()
    # Load the image data
    X_img_te = load_image_data(ID)
    return ID, X_img_te

print('Loading the training data...')
(X_img_tr, y_tr), (X_img_val, y_val) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)
print('Training data loaded!')




from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img

# A little hacky piece of code to get access to the indices of the images
# the data augmenter is working with.
class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            # We changed index_array to self.index_array
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

print('Creating Data Augmenter...')
imgen = ImageDataGenerator2(
    rotation_range=25,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zca_whitening=False,
    channel_shift_range=0.01,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr_cat, seed=np.random.randint(1, 10000))
print('Finished making data augmenter...')



from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, ZeroPadding2D
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def combined_model():

    #Get back the convolutional part of a VGG network trained on ImageNet
	model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
	model_vgg16_conv.summary()
	for layer in model_vgg16_conv.layers:
		layer.trainable = False
        

	#Create your own input format (here 3x200x200)
	#input = Input(shape=(3,200,200),name = 'image_input')
	input = Input(shape=(max_img_dim, max_img_dim, 3), name='image_input')

	#Use the generated model 
	output_vgg16_conv = model_vgg16_conv(input)

	#Add the fully-connected layers 
	x = Flatten(name='flatten')(output_vgg16_conv)
	x = Dense(128, activation='relu', name='fc1')(x)
	x = Dropout(0.5)(x)
	x = Dense(64, activation='relu', name='fc2')(x)
	x = Dropout(0.5)(x)
	x = Dense(8, activation='softmax', name='predictions')(x)
	
	#Create your own model 
	my_model = Model(input=input, output=x)

	#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
	my_model.summary()

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
	#for layer in my_model.layers[:2]:
		#layer.trainable = False
        
    # compile the model with a SGD/momentum optimizer
	my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9))

	return my_model

print('Creating the model...')
model = combined_model()
print('Model created!')



from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def combined_generator(imgen, y):
    """
    A generator to train our keras neural network. It
    takes the image augmenter generator and the array
    of the pre-extracted features.
    It yields a minibatch and will run indefinitely
    """
    while True:
        for i in range(y.shape[0]):
            # Get the image batch and labels
            batch_img, batch_y = next(imgen)
            # This is where that change to the source code we
            # made will come in handy. We can now access the indicies
            # of the images that imgen gave us.
            #x = X[imgen.index_array]
            yield [batch_img], batch_y

# autosave best Model
best_model_file = "fishnetcol.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

print('Training model...')
history = model.fit_generator(combined_generator(imgen_train, y_tr),
                              samples_per_epoch=y_tr.shape[0],
                              nb_epoch=50,
                              validation_data=([X_img_val], y_val_cat),
                              nb_val_samples=y_val.shape[0],
                              verbose=2,
                              callbacks=[best_model])

print('Loading the best model...')
model = load_model(best_model_file)
print('Best Model loaded!')




# Get the names of the column headers
LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())
LABELS[0] = "image"

index, X_img_te = load_test_data()

yPred_proba = model.predict([X_img_te])

# Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

print('Creating and writing submission...')
fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())
fp.close()
print('Finished writing submission')
# Display the submission
yPred.tail()













