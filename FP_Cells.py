import numpy as np
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt
import random
import cv2
############## Unnecessary Warning Messages###############
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from plot_keras_history import show_history, plot_history
##########################################################

batch_size = 32
img_height = 20
img_width = 20

DB_dir_train = pathlib.Path(r"/home/carlos/Desktop/Decimo/Titulacion_II/Material/FP_Cells/DB_3/train") # get data base path
DB_dir_val = pathlib.Path(r"/home/carlos/Desktop/Decimo/Titulacion_II/Material/FP_Cells/DB_3/validation") # get data base path
image_count_train = len(list(DB_dir_train.glob("*/*g"))) # get dat base size
image_count_val = len(list(DB_dir_val.glob("*/*g"))) # get dat base size
print("\nTraining Images Data Base Size: ", image_count_train, "\n")
print("Validation Images Data Base Size: ", image_count_val, "\n")
class_names = list(sorted([item.name for item in DB_dir_train.glob('*')]))
print(class_names, "\n")

datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory(DB_dir_train, target_size=(img_height, img_width), shuffle=True, batch_size=batch_size, class_mode='sparse')
val = datagen.flow_from_directory(DB_dir_val, target_size=(img_height, img_width), shuffle=False, batch_size=batch_size, class_mode='sparse')

print("\n")

e=100

model = models.Sequential()
model.add(layers.Conv2D(10, (10,10), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D(2))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
history = model.fit(train, epochs=e)
print(history.history.keys())
show_history(history)
plot_history(history, path="PF_Cells.png")

plt.close()

# print("Loading Model")
# model = tf.keras.models.load_model("/home/carlos/Desktop/Decimo/Titulacion_II/Material/FP_Categories_Classification/FP_Categories_Classification_1.4_V6_acc_0.91_epo_500")

loss, accuracy = model.evaluate(val)
print("Loss: ",loss)
print("Accuracy: ",accuracy)

model.save("FP_Cells"+str(accuracy)+"_epo_" + str(e)) #save model

# Uncomment the follow code lines if you want to use the trained mode-------------------------------

# predictions = model.predict(val)
# image1 = cv2.imread( "/home/carlos/Desktop/Decimo/Titulacion_II/FP_Categories_Classification/DB_4/train/4/train_4_50.jpg")
# print(image1.shape)
# normalize = image1 /255.0
# reshaped = np.reshape(normalize,(1,250,250,3))
# predictions = model.predict(reshaped)

# print(np.max(predictions)*100)
# print(Label[np.argmax(predictions)])

# l1 = list(range(len(val)-1))
# random.shuffle(l1)
# l2 = list(range(30))
# random.shuffle(l2)

# for i in l2:
#     for b in l1:
#         print("Label: " + str(val[b][1][i]))
#         print("Prediction: " + str(Label[np.argmax(predictions[i+b*batch_size])]) + " ------> accuracy: " + str(np.max(predictions[i+b*batch_size])*100))
#         plt.figure()
#         plt.imshow(val[b][0][i])
#         plt.title("Result \n Prediction: " + str(Label[np.argmax(predictions[i+b*batch_size])]) + " | " + str(round(np.max(predictions[i+b*batch_size])*100, 3)) + " % of acurracy \n\n" + "Label: " + str(Label[int(val[b][1][i])]))
#         plt.show()
