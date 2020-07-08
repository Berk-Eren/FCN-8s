from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, MaxPool2D, Dropout, \
                                        Activation, Input, Add, Conv2DTranspose, Cropping2D, Softmax  

from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import RMSprop

from settings import NUM_CLASSES


def vgg_16():
    inp = Input(INPUT_SHAPE, name="input")

    pad = ZeroPadding2D(padding=100, name="pad1_100")(inp)
    conv1_1 = Conv2D(filters=64, kernel_size=3, activation='relu', name="conv1_1")(pad)

    pad = ZeroPadding2D(padding=1, name="pad2_1")(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name="conv1_2")(pad)
    pool_1 = MaxPool2D(pool_size=2, strides=2, name="pool_1")(conv1_2)

    pad = ZeroPadding2D(padding=1, name="pad3_1")(pool_1)
    conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_1")(pad)

    pad = ZeroPadding2D(padding=1, name="pad4_1")(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_2")(pad)
    pool_2 = MaxPool2D(pool_size=2, strides=2, name="pool_2")(conv2_2)

    pad = ZeroPadding2D(padding=1, name="pad5_1")(pool_2)
    conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_1")(pad)

    pad = ZeroPadding2D(padding=1, name="pad6_1")(conv3_1)
    conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_2")(pad)

    pad = ZeroPadding2D(padding=1, name="pad7_1")(conv3_2)
    conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_3")(pad)
    pool_3 = MaxPool2D(pool_size=2, strides=2, name="pool_3")(conv3_3)

    pad = ZeroPadding2D(padding=1, name="pad8_1")(pool_3)
    conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_1")(pad)

    pad = ZeroPadding2D(padding=1, name="pad9_1")(conv4_1)
    conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_2")(pad)

    pad = ZeroPadding2D(padding=1, name="pad10_1")(conv4_2)
    conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_3")(pad)
    pool_4 = MaxPool2D(pool_size=2, strides=2, name="pool_4")(conv4_3)

    pad = ZeroPadding2D(padding=1, name="pad11_1")(pool_4)
    conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_1")(pad)

    pad = ZeroPadding2D(padding=1, name="pad12_1")(conv5_1)
    conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_2")(pad)

    pad = ZeroPadding2D(padding=1, name="pad13_1")(conv5_2)
    conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_3")(pad)
    pool_5 = MaxPool2D(pool_size=2, strides=2, name="pool_5")(conv5_3)

    return (inp, pool_3, pool_4, pool_5) 

#Encoder Part
inp, pool_3, pool_4, pool_5 = vgg_16()

#Decoder Part
fc_6 = Conv2D(filters=4096, kernel_size=7, activation='relu', name="fc_6")(pool_5)
fc_6 = Dropout(0.5, name="drop_6")(fc_6)

fc_7 = Conv2D(filters=4096, kernel_size=1, activation='relu', name="fc_7")(fc_6)
fc_7 = Dropout(0.5, name="drop_7")(fc_7)

score_fr = Conv2D(filters=NUM_CLASSES, kernel_size=1, name="score_fr")(fc_7)
upscore2 = Conv2DTranspose(filters=NUM_CLASSES, kernel_size=4, strides=2, use_bias=False, name="upscore2")(score_fr)
score_pool4 = Conv2D(filters=NUM_CLASSES, kernel_size=1, name="score_pool4")(pool_4)
score_pool4c = Cropping2D(cropping=((5,5), (5,5)), data_format="channels_last", name="score_pool4c")(score_pool4)

fuse_pool_4 = Add(name="fuse_pool_4")([upscore2, score_pool4c])
upscore_pool4 = Conv2DTranspose(filters=NUM_CLASSES, kernel_size=4, strides=2, use_bias=False, name="upscore_pool4")(fuse_pool_4)

score_pool3 = Conv2D(filters=NUM_CLASSES, kernel_size=1, name="score_pool_3")(pool_3)
score_pool3c = Cropping2D(((9, 9), (9, 9)), name="score_pool3c")(score_pool3)

fuse_pool3 = Add(name="fuse_pool3")([upscore_pool4, score_pool3c])
upscore8 = Conv2DTranspose(filters=NUM_CLASSES, kernel_size=16, strides=8, use_bias=False, name="upscore8")(fuse_pool3)
score = Cropping2D(((30, 29), (30, 29)), name="score")(upscore8)

loss = Softmax()(score)


model = Model(inp, loss)


model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(learning_rate=0.001),
                    metrics=[MeanIoU(num_classes=NUM_CLASSES)])