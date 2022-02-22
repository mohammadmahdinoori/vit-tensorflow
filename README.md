# ViT Tensorflow
This repository contains the tensorflow implementation of the state-of-the-art vision transformers (a category of computer vision models first introduced in [An Image is worth 16 x 16 words](https://arxiv.org/abs/2010.11929)). This repository is inspired from the work of lucidrains which is [vit-pytorch](https://github.com/lucidrains/vit-pytorch). I hope you enjoy these implementations :)


# Models  
- [Vision Transformer (An Image is worth 16 x 16 words)](#vit)  
- [Convolutional Vision Transformer](#cvt)
- [Pyramid Vision Transformer V1](#pvt_v1)
- [Pyramid Vision Transformer V2](#pvt_v2)
- [DeiT (Training Data Efficient Image Transforemrs)](#headers)

# Requirements 

```bash
pip install tensorflow
```
<a name="vit"/>

# Vision Transformer 

Vision transformer was introduced in [An Image is worth 16 x 16 words](https://arxiv.org/abs/2010.11929). This model usese a Transformer encoder to classify images with pure attention and no convolution.

![](https://github.com/mohammadmahdinoori/vit-tensorflow/blob/main/images/ViT.png?raw=true)

### Usage

#### Defining the Model
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
                    dropout_rate=0.1,
                    prediction_dropout=0.3,
)
```

##### Params
- `num_classes`: int <br />
number of classes used for the final classification head
- `patch_size`: int <br />
patch_size used for the tokenization
- `num_of_patches`: int <br />
number of patches after the tokenization which is used for the positional encoding, Generally it can be computed by the following formula `(((h-patch_size)//patch_size) + 1)*(((w-patch_size)//patch_size) + 1)` where `h` is the height of the image and `w` is the width of the image. In addition, when height and width of the image are devisable by the `patch_size` the following formula can be used as well `(h//patch_size)*(w//patch_size)`
- `d_model`: int <br />
hidden dimension of the transformer encoder and the demnesion used for patch embedding
- `heads`: int <br />
number of heads used for the multi-head attention mechanism
- `num_layers`: int <br />
number of blocks in encoder transformer
- `mlp_rate`: int <br />
the rate of expansion in the feed-forward block of each transformer block (the dimension after expansion is `mlp_rate * d_model`)
- `dropout_rate`: float <br />
dropout rate used in the multi-head attention mechanism
- `prediction_dropout`: float <br />
dropout rate used in the final prediction head of the model

#### Inference

```python
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = vitClassifier(sampleInput , training=False)
print(output.shape) # (1 , 1000)
```

#### Training

```python
vitClassifier.compile(
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
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

<a name="cvt"/>

#  Convolutional Vision Transformer 

Convolutional Vision Transformer was introduced in [here](https://arxiv.org/abs/2103.15808). This model uses a hierarchical (multi-stage) architecture with convolutional embeddings in the begining of each stage. it also uses Convolutional Transformer Blocks to improve the orginal vision transformer by adding CNNs inductive bias into the architecture.

![](https://raw.githubusercontent.com/mohammadmahdinoori/vit-tensorflow/main/images/CvT.png)

### Usage

#### Defining the Model

```python
from cvt import CvT , CvTStage
import tensorflow as tf

cvtModel = CvT(
num_of_classes=1000, 
stages=[
        CvTStage(projectionDim=64, 
                 heads=1, 
                 embeddingWindowSize=(7 , 7), 
                 embeddingStrides=(4 , 4), 
                 layers=1,
                 projectionWindowSize=(3 , 3), 
                 projectionStrides=(2 , 2), 
                 ffnRate=4,
                 dropoutRate=0.1),
        CvTStage(projectionDim=192,
                 heads=3,
                 embeddingWindowSize=(3 , 3), 
                 embeddingStrides=(2 , 2),
                 layers=1, 
                 projectionWindowSize=(3 , 3), 
                 projectionStrides=(2 , 2), 
                 ffnRate=4,
                 dropoutRate=0.1),
        CvTStage(projectionDim=384,
                 heads=6,
                 embeddingWindowSize=(3 , 3),
                 embeddingStrides=(2 , 2),
                 layers=1,
                 projectionWindowSize=(3 , 3),
                 projectionStrides=(2 , 2), 
                 ffnRate=4,
                 dropoutRate=0.1)
],
dropout=0.5)
```

##### CvT Params
- `num_of_classes`: int <br />
number of classes used in the final prediction layer
- `stages`: list of CvTStage <br />
list of cvt stages
- `dropout`: float <br />
dropout rate used for the prediction head

##### CvTStage Params
- `projectionDim`: int <br />
dimension used for the multi-head attention mechanism and the convolutional embedding
- `heads`: int <br />
number of heads in the multi-head attention mechanism
- `embeddingWindowSize`: tuple(int , int) <br />
window size used for the convolutional emebdding
- `embeddingStrides`: tuple(int , int) <br />
strides used for the convolutional embedding
- `layers`: int <br />
number of convolutional transformer blocks
- `projectionWindowSize`: tuple(int , int) <br />
window size used for the convolutional projection in each convolutional transformer block
- `projectionStrides`: tuple(int , int) <br />
strides used for the convolutional projection in each convolutional transformer block
- `ffnRate`: int <br />
expansion rate of the mlp block in each convolutional transformer block
- `dropoutRate`: float <br />
dropout rate used in each convolutional transformer block

#### Inference

```python
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = cvtModel(sampleInput , training=False)
print(output.shape) # (1 , 1000)
```

#### Training

```python
cvtModel.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
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

<a name="pvt_v1"/>

#  Pyramid Vision Transformer V1 

Pyramid Vision Transformer V1 was introduced in [here](https://arxiv.org/abs/2102.12122). This model stacks multiple Transformer Encoders to form the first convolution-free multi-scale backbone for various visual tasks including Image Segmentation , Object Detection and etc. In addition to this a new attention mechanism called Spatial Reduction Attention (SRA) is also introduced in this paper to reduce the quadratic complexity of the multi-head attention mechansim.

![](https://raw.githubusercontent.com/mohammadmahdinoori/vit-tensorflow/main/images/PvT%20V1.png)

### Usage

#### Defining the Model

```python
from pvt_v1 import PVT , PVTStage
import tensorflow as tf

pvtModel = PVT(
num_of_classes=1000, 
stages=[
        PVTStage(d_model=64,
                 patch_size=(2 , 2),
                 heads=1,
                 reductionFactor=2,
                 mlp_rate=2,
                 layers=2, 
                 dropout_rate=0.1),
        PVTStage(d_model=128,
                 patch_size=(2 , 2),
                 heads=2, 
                 reductionFactor=2, 
                 mlp_rate=2, 
                 layers=2, 
                 dropout_rate=0.1),
        PVTStage(d_model=320,
                 patch_size=(2 , 2),
                 heads=5, 
                 reductionFactor=2, 
                 mlp_rate=2, 
                 layers=2, 
                 dropout_rate=0.1),
],
dropout=0.5)
```

##### PVT Params
- `num_of_classes`: int <br />
number of classes used in the final prediction layer
- `stages`: list of PVTStage <br />
list of pvt stages
- `dropout`: float <br />
dropout rate used for the prediction head

##### PVTStage Params
- `d_model`: int <br />
dimension used for the `SRA` mechanism and the patch embedding
- `patch_size`: tuple(int , int) <br />
window size used for the patch emebdding
- `heads`: int <br />
number of heads in the `SRA` mechanism
- `reductionFactor`: int <br />
reduction factor used for the down sampling of the `K` and `V` in the `SRA` mechanism
- `mlp_rate`: int <br />
expansion rate used in the feed-forward block
- `layers`: int <br />
number of transformer encoders
- `dropout_rate`: float <br />
dropout rate used in each transformer encoder

#### Inference

```python
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = pvtModel(sampleInput , training=False)
print(output.shape) # (1 , 1000)
```

#### Training

```python
pvtModel.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
                 tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5 , name="top_5_accuracy"),
        ])

pvtModel.fit(
        trainingData, #Tensorflow dataset of images and labels in shape of ((b , h , w , 3) , (b,))
        validation_data=valData, #The same as training
        epochs=100,)
```

<a name="pvt_v2"/>

#  Pyramid Vision Transformer V2

Pyramid Vision Transformer V2 was introduced in [here](https://arxiv.org/abs/2106.13797). This model is an improved version of the PVT V1 which has three improvements: <br />
1) It uses overlapping patch embedding by using padded convolutions
2) It uses convolutional feed-forward blocks which have a depth-wise convolution after the first fully-connected layer
3) It uses a fixed pooling instead of convolutions for down sampling the K and V in the SRA attention mechanism (The new attention mechanism is called Linear SRA)

![](https://raw.githubusercontent.com/mohammadmahdinoori/vit-tensorflow/main/images/PvT%20V2.png)

### Usage

#### Defining the Model

```python
from pvt_v2 import PVTV2 , PVTV2Stage
import tensorflow as tf

pvtV2Model = PVTV2(
num_of_classes=1000, 
stages=[
        PVTV2Stage(d_model=64,
                   windowSize=(2 , 2), 
                   heads=1,
                   poolingSize=(7 , 7), 
                   mlp_rate=2, 
                   mlp_windowSize=(3 , 3), 
                   layers=2, 
                   dropout_rate=0.1),
        PVTV2Stage(d_model=128, 
                   windowSize=(2 , 2),
                   heads=2,
                   poolingSize=(7 , 7), 
                   mlp_rate=2, 
                   mlp_windowSize=(3 , 3), 
                   layers=2,
                   dropout_rate=0.1),
        PVTV2Stage(d_model=320,
                   windowSize=(2 , 2), 
                   heads=5, 
                   poolingSize=(7 , 7), 
                   mlp_rate=2, 
                   mlp_windowSize=(3 , 3), 
                   layers=2, 
                   dropout_rate=0.1),
],
dropout=0.5)
```

##### PVT Params
- `num_of_classes`: int <br />
number of classes used in the final prediction layer
- `stages`: list of PVTV2Stage <br />
list of pvt v2 stages
- `dropout`: float <br />
dropout rate used for the prediction head

##### PVTStage Params
- `d_model`: int <br />
dimension used for the `Linear SRA` mechanism and the convolutional patch embedding
- `windowSize`: tuple(int , int) <br />
window size used for the convolutional patch emebdding
- `heads`: int <br />
number of heads in the `Linear SRA` mechanism
- `poolingSize`: tuple(int , int) <br />
size of the K and V after the fixed pooling
- `mlp_rate`: int <br />
expansion rate used in the convolutional feed-forward block
- `mlp_windowSize`: tuple(int , int) <br />
the window size used for the depth-wise convolution in the convolutional feed-forward block
- `layers`: int <br />
number of transformer encoders
- `dropout_rate`: float <br />
dropout rate used in each transformer encoder

#### Inference

```python
sampleInput = tf.random.normal(shape=(1 , 224 , 224 , 3))
output = pvtV2Model(sampleInput , training=False)
print(output.shape) # (1 , 1000)
```

#### Training

```python
pvtV2Model.compile(
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
          metrics=[
                   tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                   tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5 , name="top_5_accuracy"),
          ])

pvtV2Model.fit(
          trainingData, #Tensorflow dataset of images and labels in shape of ((b , h , w , 3) , (b,))
          validation_data=valData, #The same as training
          epochs=100,)
```

