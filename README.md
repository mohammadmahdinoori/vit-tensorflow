# vit-tensorflow 
hello


# Models 
**Table of Contents**

[TOCM]

[TOC]

- ###### Vision Transformer (An Image is worth 16 x 16 words)
- ###### Pyramid Vision Transformer
- ###### Pyramid Vision Transforemr V2
- ###### Convolutional Vision Transformer
- ###### DeiT (Training Data Efficient Image Transforemrs)

# Requirements 

```bash
pip install tensorflow
```

# Vision Transformer 
                
```python
from vit import ViT
import tensorflow as tf

vitClassifier = ViT(
                    num_classes=1000,
                    patch_size=16,
                    num_of_patches=(224//16)**2,
                    d_model=128,
                    heads=2,
                    num_layers=4,
                    mlp_rate=2,
                    dropout_rate=0.1
)

#inference
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = vitClassifier(sampleInput , training=False)
print(output.shape) # (1 , 1000)

#training
vitClassifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=[
                       tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5 , name="top_5_accuracy"),
              ])

vitClassifier.fit(
trainingData, #Tensorflow dataset of images and labels in shape of ((b , h , w , 3) , (b,))
validation_data=valData, #The same as training
epochs=100,)
```

#  Convolutional Vision Transformer 
                
```python
from cvt import CvT , CvTStage
import tensorflow as tf

cvtModel = CvT(1000 , [
                      CvTStage(projectionDim=64, 
                               heads=1, 
                               embeddingWindowSize=(7 , 7), 
                               embeddingStrides=(4 , 4), 
                               layers=1,
                               projectionWindowSize=(3 , 3), 
                               projectionStrides=(2 , 2), 
                               ffnRate=4),
                      CvTStage(projectionDim=192,
                               heads=3,
                               embeddingWindowSize=(3 , 3), 
                               embeddingStrides=(2 , 2),
                               layers=1, 
                               projectionWindowSize=(3 , 3), 
                               projectionStrides=(2 , 2), 
                               ffnRate=4),
                      CvTStage(projectionDim=384,
                               heads=6,
                               embeddingWindowSize=(3 , 3),
                               embeddingStrides=(2 , 2),
                               layers=1,
                               projectionWindowSize=(3 , 3),
                               projectionStrides=(2 , 2), 
                               ffnRate=4)
])

#inference
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = cvtModel(sampleInput , training=False)
print(output.shape) # (1 , 1000)

#training
cvtModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=[
                       tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5 , name="top_5_accuracy"),
              			 ])

cvtModel.fit(
trainingData, #Tensorflow dataset of images and labels in shape of ((b , h , w , 3) , (b,))
validation_data=valData, #The same as training
epochs=100,)
```
