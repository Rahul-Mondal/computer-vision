
import numpy as np
import multiprocessing

from keras import metrics
from keras import optimizers
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal, Constant
from keras.constraints import maxnorm
from keras.utils import np_utils

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Initialize network parameters
batch_size = 64
epochs = 50
num_classes = 2

def load_data():
	
	print('Loading data . . .')
	
	i = np.load('x_labels.npy')
	print("\nNumber of samples: %s" % len(i))
	
	j = i.reshape(len(i), 60, 60, 3)
	k = np.load('y_labels.npy')
		  
	return (j, k)

# Load data and normalize
X, y = load_data()

# Change type and Normalize
X = X.astype('float32')
X /= 255

# 1-hot encoding
y = np_utils.to_categorical(y, num_classes)

print('\nSplitting data into train/val/test. . ')
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=65)

print('\nTraining: %s  Validation: %s  Testing: %s' % (len(X_train),len(X_val),len(X_test)))

print('\nshuffling data . . ')
# shuffle the data before training/testing	
X_train, y_train = shuffle(X_train, y_train, random_state=73)
X_val, y_val = shuffle(X_val, y_val, random_state=73)
X_test, y_test = shuffle(X_test, y_test, random_state=73)

# Model
model = Sequential()

def define_model():

	print('\n defining model . . .')	
	gaussian = RandomNormal(mean=0., stddev=0.1)
	cons = Constant(value=2.)

	model.add(Conv2D(filters=32, kernel_size=(7, 7), padding='same', kernel_initializer=gaussian,
		use_bias=True, bias_initializer=cons, bias_constraint=maxnorm(5.),input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Conv2D(filters=64, kernel_size=(5, 5)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(filters=256, kernel_size=(5, 5)))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
	model.add(MaxPooling2D(pool_size=(3,3)))

	model.add(Flatten())
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

def compile_model():

	print('\ncompiling model')
	
	# compile the model
	ada = optimizers.Adagrad(lr=0.0001, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['accuracy', metrics.binary_accuracy, 'mae'])

# Train the model

def train_model():
	print('\npreprocessing data for training . .')

	# preprocessers for the training/validation data	
	train_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True, zca_whitening=False)
	val_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True, zca_whitening=False)
	train_datagen.fit(X_train)
	val_datagen.fit(X_val)

	print('\nstart training . .')

	# fits the model on batches with real-time data preprocessing:
	model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),
						steps_per_epoch = len(X_train) / batch_size,
						validation_data = val_datagen.flow(X_val, y_val, batch_size=batch_size),
						validation_steps = len(X_val) / batch_size,
						callbacks = get_callbacks(),
						epochs = epochs,
						verbose = 1)
	
	# save the model to disk    
	save_model(model)

def get_callbacks():
	
	# training logger callback
	csv = CSVLogger('training.log', separator=',')

	# learning rate reducer callback
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

	# early stopping callback
	es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto')

	# model checkpoint callback
	path='checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
	mcp = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=5)

	return [csv, reduce_lr, mcp]	
	
def save_model(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open('model.json', "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("\nSaved model to disk")

def load_model():
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("\nLoaded model from disk")
	return loaded_model

def run(pool):

	define_model()
	compile_model()
	train_model()

	print('\nTraining over !')
	

'''

def evaluate():
	# Evaluate the results for test set
	model = load_model('model.json')
	# Compile the model
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	score = model.evaluate(X_test, y_test, verbose=1)

	print("Score: {}  |  Accuracy: {}".format(score[0], score[1]))
'''

def call_with_multiprocessing_pool(func):
	n_cpus = multiprocessing.cpu_count()
	print("multiprocessing: using %s processes" % n_cpus)
	pool = multiprocessing.Pool(n_cpus)
	func(pool)
	pool.close()
	pool.join()

if __name__ == "__main__":
    
	call_with_multiprocessing_pool(run)



