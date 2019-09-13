import os
import cv2
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt
from librosa import display
from keras.utils import to_categorical
from librosa.feature import melspectrogram
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

import json
from keras.models import model_from_json

os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

class Test:
	@staticmethod
	def data_importer(directory, categories, height, width):
		X_test, y_test = [], []
		for category in categories:
			path = os.path.join(directory, category)
			class_num = categories.index(category)
			try:
				for img in os.listdir(path):
					image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
					new_array = cv2.resize(image_array, (height, width))
					X_test.append(new_array)
					y_test.append(class_num)
			except Exception as e:
				pass
		# return data
		X_test = np.asarray(X_test)
		return X_test, y_test
	
	@staticmethod
	def label_encoding(y_test):
		label_encoder = LabelEncoder()
		encoded_y = label_encoder.fit_transform(y_test)
		one_hot_y_test = to_categorical(encoded_y)

		return one_hot_y_test


if __name__ == '__main__':

	# working_directory = os.getcwd()
	# training_directory = os.path.join(working_directory, 'test')
	# # print(os.listdir(training_directory))
	# CLASSES = ['BacterialLeafBlight', 'BrownSpot', 'Healthy', 'Hispa', 'LeafBlast', 'LeafSmut']
	
	# test = Test()

	# X_test, y_test = test.data_importer(training_directory, CLASSES, 120, 120)
	# print(X_test[1])
	# print(y_test[1])
	# y_test = test.label_encoding(y_test)


	# X_test = np.asarray(X_test)
	# print(X_test.shape)
	
	

	# # print('X_test[1]:')
	# # print(X_test[1])

	# # print('y_test[1]:')
	# # print(y_test[1])
	# X_train = np.load('X_train_12000_80.npy')
	# y_train = np.load('y_train_12000_80.npy')


	# meu = np.mean(X_train)
	# sigma_2 = np.std(X_train)
	# X_train = (X_train - meu) / sigma_2
	# X_test = (X_test - meu) / sigma_2
	

	# np.save("X_test_120_120_normalized.npy", X_test)
	# np.save("y_test_120_120_normalized.npy", y_test)


	X_test = np.load("X_test_120_120_normalized.npy")
	y_test = np.load("y_test_120_120_normalized.npy")	
	

	#converting to array
	# print(X_test.shape)
	# print(y_test.shape)

	# load json and create model
	json_file = open('cnn_model_warmup_adam_reduce_lr_12000_80_normalized.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("cnn_model_warmup_adam_reduce_lr_12000_80_normalized.h5")
	print("Loaded model from disk")



	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	score = loaded_model.evaluate(X_test, y_test, verbose=1)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
	

	prediction = loaded_model.predict_classes(X_test)
	y_test_non_category = [np.argmax(t) for t in y_test]
	conf_mat = confusion_matrix(y_test_non_category, prediction) 
	print(conf_mat)
	plt.plot(conf_mat)
	plt.show()

	# for pred in prediction:
		
	# 	if pred == 0:
	# 		print('Angry')
	# 	elif pred == 1:
	# 		print('Disgust')
	# 	elif pred == 2:
	# 		print('Fear')
	# 	elif pred == 3:
	# 		print('Happy')
	# 	elif pred == 4:
	# 		print('Neutral')
	# 	elif pred == 5:
	# 		print('Surprise')
	# 	elif pred == 6:
	# 		print('Sad')

	# print(prediction)