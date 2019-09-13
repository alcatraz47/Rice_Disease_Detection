import os
import cv2
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input

from keras_radam import RAdam

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('cnn_model_warmup_adam_reduce_lr_12000_80_normalized.png')
    plt.show()


class Train:
	@staticmethod
	def data_importer(directory, categories, height, width):
		X_train, y_train = [], []
		X_val, y_val = [], []
		counter = 1
		print('importing data.', end='')
		for category in categories:
			path = os.path.join(directory, category)
			class_num = categories.index(category)
			try:
				# print('.', end='')
				for img in os.listdir(path):
					image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
					new_array = cv2.resize(image_array, (height, width))
					# plt.imshow(new_array)
					# data.append([new_array, class_num])
					if counter == 10:
						X_val.append(new_array)
						y_val.append(class_num)
						counter = 1
						continue
					X_train.append(new_array)
					y_train.append(class_num)
					counter+=1
			except Exception as e:
				pass
		# return data
		X_train = np.asarray(X_train)
		X_val = np.asarray(X_val)
		return X_train, y_train, X_val, y_val
	
	@staticmethod
	def feature_label_extractor(data, X, y):
		for feature, label in data:
			X.append(feature)
			y.append(label)
	
	@staticmethod
	def label_encoding(y_train, y_val):
		label_encoder = LabelEncoder()
		encoded_y = label_encoder.fit_transform(y_train)
		one_hot_y_train = to_categorical(encoded_y)

		label_encoder = LabelEncoder()
		encoded_y = label_encoder.fit_transform(y_val)
		one_hot_y_val = to_categorical(encoded_y)		
		return one_hot_y_train, one_hot_y_val

	@staticmethod
	def train_model(input_shape, classes):
		model = Sequential()
		#1st cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		model.add(Conv2D(16, (3,3), padding = "same", input_shape = input_shape, kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(Conv2D(16, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(MaxPool2D(2,2))
		model.add(Dropout(0.25))

		#2nd cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		model.add(Conv2D(16, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(Conv2D(16, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(MaxPool2D(2,2))
		model.add(Dropout(0.25))

		#     #3rd cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		#     model.add(Conv2D(16, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(Conv2D(16, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(MaxPool2D(2,2))
		#     model.add(Dropout(0.25))

		#4th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		model.add(Conv2D(32, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(Conv2D(32, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(MaxPool2D(2,2))
		model.add(Dropout(0.25))

		#     #5th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		#     model.add(Conv2D(32, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(Conv2D(32, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(MaxPool2D(2,2))
		#     model.add(Dropout(0.25))

		#     #6th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		#     model.add(Conv2D(32, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(Conv2D(32, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(MaxPool2D(2,2))
		#     model.add(Dropout(0.25))

		#7th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		#     model.add(Conv2D(64, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(Conv2D(64, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		#     model.add(Activation("relu"))
		#     model.add(BatchNormalization(axis = -1))
		#     model.add(MaxPool2D(2,2))
		#     model.add(Dropout(0.25))

		#8th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		model.add(Conv2D(64, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(Conv2D(64, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(MaxPool2D(2,2))
		model.add(Dropout(0.25))

		#9th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		model.add(Conv2D(128, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(Conv2D(128, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(MaxPool2D(2,2))
		model.add(Dropout(0.25))

		#10th cnn -> relu -> batch_norm -> cnn -> relu -> batch_norm -> maxpool -> dropout
		model.add(Conv2D(128, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(Conv2D(128, (3,3), padding = "same", kernel_regularizer = regularizers.l2(.01)))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = -1))
		model.add(MaxPool2D(2,2))
		model.add(Dropout(0.25))

		#fcn flatten -> dense 64 -> relu -> batch_norm -> droput -> dense (classes) -> softmax
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(.5))

		#softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model


if __name__ == '__main__':

	working_directory = os.getcwd()
	print (working_directory)

	# X_train, y_train = [], []

	train = Train()

	CLASSES = ['BacterialLeafBlight', 'BrownSpot', 'Healthy', 'Hispa', 'LeafBlast', 'LeafSmut']
	# X_train, y_train, X_val, y_val = train.data_importer(working_directory, CLASSES, 120, 120)
	# # random.shuffle(training_data)
	# # train.feature_label_extractor(training_data, X_train, y_train)
	# # X_train = np.array(X_train).reshape(len(X_train), 120, 120, 3)
	# # X_val = np.array(X_train).reshape(len(X_val), 120, 120, 3)

	# y_train, y_val = train.label_encoding(y_train, y_val)


	# np.save('X_train_12000_80.npy', X_train)
	# np.save('y_train_12000_80.npy', y_train)

	# np.save('X_val_12000_80.npy', X_val)
	# np.save('y_val_12000_80.npy', y_val)

	X_train = np.load('X_train_12000_80.npy')
	y_train = np.load('y_train_12000_80.npy')

	X_val = np.load('X_val_12000_80.npy')
	y_val = np.load('y_val_12000_80.npy')

	meu = np.mean(X_train)
	sigma_2 = np.std(X_train)
	X_train = (X_train - meu) / sigma_2
	X_val = (X_val - meu) / sigma_2

	print(X_train.shape)
	print(X_val.shape)

	model = train.train_model((120, 120, 3), 6)

	epochs = 100
	batch_size = 32
	initial_learning_rate = 1e-4
	optim = Adam(lr = initial_learning_rate, decay = initial_learning_rate / epochs)
	# optim = RAdam(warmup_proportion = 0.1, min_lr = 1e-5)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_lr=0.01)
	model.summary()
	

	model.compile(optimizer = optim, loss = 'categorical_crossentropy', metrics = ['accuracy'])
	model_info = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks = [reduce_lr])
	plot_model_history(model_info)

	model_json = model.to_json()
	with open("cnn_model_warmup_adam_reduce_lr_12000_80_normalized.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("cnn_model_warmup_adam_reduce_lr_12000_80_normalized.h5")
	print("Saved model to disk")