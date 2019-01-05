# Document Classification based on Writing style

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/671a7ec12ff04b378bfd7605d7478a2b)](https://app.codacy.com/app/adelra/document_style_classification?utm_source=github.com&utm_medium=referral&utm_content=adelra/document_style_classification&utm_campaign=Badge_Grade_Dashboard)

This repository uses CNNs for classifying documents. The network has 4m parameters and uses 3, 64 layer 1d convolutional layers. Details are:
```
Layer (type)                 Output Shape              Param #
=================================================================
conv1d_1 (Conv1D)            (None, 73971, 64)         384
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 36984, 64)         0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 36984, 64)         20544
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 7396, 64)          0
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 7396, 64)          20544
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 1479, 64)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 94656)             0
_________________________________________________________________
dense_1 (Dense)              (None, 50)                4732850
_________________________________________________________________
softmax (Dense)              (None, 4)                 204
=================================================================
Total params: 4,774,526
Trainable params: 4,774,526
Non-trainable params: 0
_________________________________________________________________
```

As you can see one of the layers that has the biggest number of parameters is the first FC layer. Basically, we try to avoid FC connections because they don't do much on the network. Also, in the layers con2 and conv3 you can see that both layers have the same number of parameters but their shape is different. This shows that reducing the heights and weights and adding to depth can help much in the terms of memory.  
