#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importar librerias
import numpy as np
import pandas as pd
import os
from skimage import io
import matplotlib.pyplot as plt  
import re


# In[ ]:


#cargar las imagenes de la carpeta de Frutas
dirname = os.path.join(os.getcwd(), 'train')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search(r"\.(jpg|jpeg|png)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))
#images[1]



# In[3]:


etiquetas=[]
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        nombre = filename.split("_")[0]   # todo antes del "_"
        etiquetas.append(nombre)

print(etiquetas)


# In[4]:


#print(type(images))
#print(type(images[0]))
#print(images[0].shape)


# In[5]:


#array de imagénes
images_array = np.array(images, dtype=object)


# In[6]:


from skimage.transform import resize

def simple_resize(images, new_size=(224, 224)):
    resized = []
    for img in images:
        if img.ndim == 2:
            r = resize(img, new_size, preserve_range=True, anti_aliasing=True)
        else:
            r = resize(img, (new_size[0], new_size[1], img.shape[2]), preserve_range=True, anti_aliasing=True)

        resized.append(r.astype(img.dtype))

    return resized  


# In[7]:


resized_images = simple_resize(images_array, new_size=(224 , 224))


# ### Preprocesado de imágenes ###

# In[8]:


# Mostrar las imagenes descargadas
'''
for image in resized_images:
    fig, (ax) = plt.subplots(1)
    fig.set_figwidth(15)
    ax.imshow(image)

'''



# In[9]:


#Transformar rojo , verde  y azul
imagesRGB =[]
for image in resized_images:
    image_rgb = image[:, :, [0, 1, 2]]  
    imagesRGB.append(image_rgb)


# In[10]:


#Mostrar imágenes en RGB
'''
for image in imagesRGB:
    fig, (ax) = plt.subplots(1)
    fig.set_figwidth(15)
    ax.imshow(image)
'''


# In[11]:


#normalizar imagenes
X = np.array(imagesRGB)
print("Shape:", X.shape)       # debería decir (n_imágenes, 244, 244, 3)
print("Tipo:", X.dtype)        # probablemente uint8
print("Rango original:", X.min(), "a", X.max())
X = X.astype('float32') / 255.0 



# In[12]:


import numpy as np
import matplotlib.pyplot as plt

print("----- COMPROBACIÓN DE LA NORMALIZACIÓN ----")
print(f"Shape: {X.shape}")
print(f"Tipo de dato: {X.dtype}")
print(f"Valor mínimo: {X.min():.5f}")
print(f"Valor máximo: {X.max():.5f}")


# In[13]:


from sklearn.preprocessing import LabelEncoder
import numpy as np

encoder = LabelEncoder()
Y = encoder.fit_transform(etiquetas)


# In[14]:


import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


# In[15]:


import tensorflow as tf
from tensorflow.keras import layers, models

num_clases = len(np.unique(Y))

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_clases, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#model.summary()


# In[16]:


history = model.fit(
    X, Y,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=True
)


# In[20]:


model.save("modelo_frutas.h5")


# In[21]:


import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

model = tf.keras.models.load_model("modelo_frutas.h5")
encoder = pickle.load(open("label_encoder.pkl", "rb"))

def predecir(ruta):
    img = Image.open(ruta).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  

    pred = model.predict(img)
    clase = np.argmax(pred)
    etiqueta = encoder.inverse_transform([clase])[0]

    print("Predicción:", etiqueta)



# In[25]:


dirname = os.path.join(os.getcwd(),"image.png")
imgpath = dirname 

predecir(imgpath)


# In[27]:


dirname = os.path.join(os.getcwd(),"image2.jpg")
imgpath = dirname 

predecir(imgpath)

