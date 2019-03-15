import numpy as np
import torchvision.models as models
from keras.applications.vgg16 import VGG16

pytorch_model = models.vgg16(pretrained=True)

# select weights in the conv2d layers and transpose them to keras dim ordering:
wblist_torch = list(pytorch_model.parameters())[:26]
wblist_keras = []
for i in range(len(wblist_torch)):
    if wblist_torch[i].dim() == 4:
        w = np.transpose(wblist_torch[i].detach().numpy(), axes=[2, 3, 1, 0])
        wblist_keras.append(w)
    elif wblist_torch[i].dim() == 1:
        b = wblist_torch[i].detach().numpy()
        wblist_keras.append(b)
    else:
        raise Exception('Fully connected layers are not implemented.')

keras_model = VGG16(include_top=False, weights=None)
keras_model.set_weights(wblist_keras)
keras_model.save_weights('vgg16_pytorch2keras.h5')
