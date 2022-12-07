#efficientnet
import numpy as np 


from keras import layers
from keras.preprocessing.image import ImageDataGenerator 

import efficientnet.tfkeras as efn
from tensorflow.keras import models

import os

# import callbacks
from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Plotting functions
import itertools
import matplotlib.pyplot as plt

#metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
#%%%

#Creating the paths 
train_dir = 'C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/Railway Track fault Detection Updated/Train'
val_dir   = 'C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/Railway Track fault Detection Updated/Validation'
test_dir  = 'C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/Railway Track fault Detection Updated/Test'

train_defective_fnames = os.listdir(train_dir+'/Defective' )
train_nondefective_fnames = os.listdir(train_dir+'/Non defective')



#%%%

target_size=(300,300)
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=False,    
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=False,     
    class_mode=None)

#%%%
num_classes = 2
input_shape = (300,300,3)
#%%

base_model = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)
base_model.trainable = False # freeze the base model (for transfer learning)

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128)(x)
out = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=out)

model.summary()

#%%%

# compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#%%%
step_train=train_generator.n//train_generator.batch_size
step_val  =val_generator.n//val_generator.batch_size
num_epochs = 30
#%%
# Train Model
filepath = "C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/saved models/efficientnet.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train Model
history = model.fit(train_generator,steps_per_epoch= step_train,epochs=num_epochs,callbacks = callbacks_list, validation_data=val_generator, validation_steps=step_val) 
#%%%
#Visualizing training performance
plt.figure(figsize=(12, 8))
epochs= 30

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.grid()
plt.xticks(np.arange(0, epochs+1, step=4))
plt.ylabel("Score")
plt.xlabel("Epochs")
plt.title('Loss')

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.grid()
plt.xticks(np.arange(0, epochs+1, step=4))
plt.ylabel("Score")
plt.xlabel("Epochs")
plt.title('Accuracy')
plt.show()
#%%%
model = load_model("C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/saved models/efficientnet.hdf5")

predictions = model.predict(test_generator)     # Vector of probabilities


pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability

true_labels = test_generator.classes 
#%%%
"""
All plotting functions now put together at the top of the doc
"""

def plot_confusion_matrix(cm,classes,title='Confusion Matrix',cmap=plt.cm.Blues):
    
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
        pass
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    pass


# Imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ROC Function
def plot_roc_curve(Y_test, model_probs):

	random_probs = [0 for _ in range(len(Y_test))]
	# calculate AUC
	model_auc = roc_auc_score(Y_test, model_probs)
	# summarize score
	print('Model: ROC AUC=%.3f' % (model_auc))
	# calculate ROC Curve
		# For the Random Model
	random_fpr, random_tpr, _ = roc_curve(Y_test, random_probs)
		# For the actual model
	model_fpr, model_tpr, _ = roc_curve(Y_test, model_probs)
	# Plot the roc curve for the model and the random model line
	plt.plot(random_fpr, random_tpr, linestyle='--', label='Random')
	plt.plot(model_fpr, model_tpr, marker='.', label='Model')
	# Create labels for the axis
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()
    

#%%%
cnf_mat = confusion_matrix(true_labels,pred_labels)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_mat,classes= ['Defective', 'Non Defective'])
plt.show()

#%%%
report=classification_report(true_labels, pred_labels, target_names=['Defective', 'Non Defective'])
print(report)
#%%%
plot_roc_curve(true_labels,pred_labels)
#%%%
model2 = load_model("C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/saved models/inception.hdf5")

predictions = model2.predict(test_generator)     # Vector of probabilities


pred_labels2 = np.argmax(predictions, axis = 1) # We take the highest probability

true_labels2 = test_generator.classes 

#%%%
model3 = load_model("C:/Users/TAbos/OneDrive/Documenten/University Projects/Master Thesis/saved models/resnet.hdf5")

predictions = model3.predict(test_generator)     # Vector of probabilities


pred_labels3 = np.argmax(predictions, axis = 1) # We take the highest probability

true_labels3 = test_generator.classes 
#%%%
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(true_labels2,pred_labels2)
auc = metrics.roc_auc_score(true_labels2,pred_labels2)
plt.plot(fpr,tpr,label="Inception, auc={}".format(round((auc),2)))

fpr, tpr, thresh = metrics.roc_curve(true_labels3,pred_labels3)
auc = metrics.roc_auc_score(true_labels3,pred_labels3)
plt.plot(fpr,tpr,label="ResNet, auc={}".format(round((auc),2)))

fpr, tpr, thresh = metrics.roc_curve(true_labels,pred_labels)
auc = metrics.roc_auc_score(true_labels,pred_labels)
plt.plot(fpr,tpr,label="EfficientNet, auc={}".format(round((auc),2)))

plt.legend(loc=0)