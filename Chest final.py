#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import util


# In[2]:


train_df = pd.read_csv("train-small.csv")
valid_df = pd.read_csv("valid-small.csv")

test_df = pd.read_csv("test.csv")

train_df.head()


# In[3]:


labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']


# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li> Make use of python's set.intersection() function. </li>
# </ul>
# </p>

# In[4]:


def check_for_leakage(df1, df2, patient_col):
    
    df1_patients_unique = set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    leakage = len(patients_in_both_groups) > 0
    
    return leakage


# In[5]:


print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))


# In[6]:


def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
     
    print("getting train generator...") 
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
  
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator


# In[7]:


def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    
    print("getting train and valid generators...")
    
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size =(target_w, target_h))
    
   
    batch = raw_train_generator.next()
    data_sample = batch[0]

    
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator


# In[8]:


IMAGE_DIR = "H:\\Chest model git\\Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning-master\\nih\\small\\"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# In[10]:


def compute_class_freqs(labels):
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


# In[11]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos


# In[13]:


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


# In[15]:


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
   
    def weighted_loss(y_true, y_pred):
    
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        
        return loss
    
        ### END CODE HERE ###
    return weighted_loss


# In[16]:


base_model = DenseNet121(weights="H:\\Chest model git\\Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning-master\\nih\\densenet.hdf5", include_top=False)

x = base_model.output

x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))


# In[17]:


model.load_weights("H:\\Chest model git\\Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning-master\\nih\\pretrained_model.h5")


# In[18]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))


# In[20]:


df = pd.read_csv("train-small.csv")
IMAGE_DIR = "H:\\Chest model git\\Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning-master\\nih\\small\\"
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])[:4]


# In[21]:


from keras.preprocessing import image

def compute_gradcam(model, img_path, image_dir, df, labels, selected_labels):
    # Load and preprocess the image
    img = image.load_img(image_dir + img_path, target_size=(320, 320))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_input = img_array / 255.0

    # Get the class predictions
    preds = model.predict(preprocessed_input)
    class_idxs_sorted = np.argsort(preds.flatten())[::-1]

    # Get the class activation map
    last_conv_layer_name = "conv5_block16_concat"
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = model.get_layer("global_average_pooling2d")(x)
    output = model.get_layer("dense")(x)
    classifier_model = tf.keras.Model(classifier_input, output)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(preprocessed_input)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize heatmap to match the original image size
    img = cv2.imread(image_dir + img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Print the predicted labels and their probabilities
    for idx in class_idxs_sorted[:14]:
        print(f'{labels[idx]}: {preds[0][idx]}')

    # Display the image with Grad-CAM heatmap
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# In[30]:


# Choose an image to visualize Grad-CAM
import tensorflow as tf
import cv2


img_path = '00027240_004.png'

selected_labels = ['Cardiomegaly', 'Effusion']  # Example labels to display Grad-CAM for

compute_gradcam(model, img_path,"H:\\Chest model git\\Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning-master\\nih\\small\\", df, labels, selected_labels)


# In[34]:


import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.models import Model

def compute_gradcam(model, img_path, image_dir, df, labels, selected_labels):
    # Load and preprocess the image
    img = image.load_img(image_dir + img_path, target_size=(320, 320))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_input = img_array / 255.0

    # Get the class predictions
    preds = model.predict(preprocessed_input)
    class_idxs_sorted = np.argsort(preds.flatten())[::-1]

    # Get the class activation map
    last_conv_layer_name = "conv5_block16_concat"
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = model.get_layer("global_average_pooling2d")(x)
    output = model.get_layer("dense")(x)
    classifier_model = tf.keras.Model(classifier_input, output)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(preprocessed_input)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize heatmap to match the original image size
    img = cv2.imread(image_dir + img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Print the predicted labels and their probabilities within the range of 0.5 to 0.8
    for idx in class_idxs_sorted:
        if 0.5 <= preds[0][idx] <= 0.8:
            print(f'{labels[idx]}: {preds[0][idx]}')

    # Display the image with Grad-CAM heatmap
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Call the function with your parameters
img_path = '00029188_001.png'
compute_gradcam(model, img_path, "H:\\Chest model git\\Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning-master\\nih\\small\\", df, labels, selected_labels)


# In[ ]:




