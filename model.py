import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class TinyssimoRadar(keras.Model):
    def __init__(self, cnn_cfg:dict,
                tcn_cfg:dict,
                class_cfg:dict,
                num_classes:int,
                return_sequence:bool=False,
                *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnn_cfg = cnn_cfg
        self.tcn_cfg = tcn_cfg
        self.class_cfg = class_cfg

        self.return_sequence = return_sequence
        self.num_classes = num_classes

        self.frame_num = None

    def build(self, input_shape):
        # print(f"TinyssimoRadar build with {input_shape}")
        self.frame_num = input_shape[1]

        self.cnn = FeatureExtractor(channels=self.cnn_cfg['channels'],
                                    groups=self.cnn_cfg['groups'],
                                    drop_rate=self.cnn_cfg['drop_rate'],
                                    kernel_regularizer=self.cnn_cfg['kernel_regularizer'],
                                    activity_regularizer=self.cnn_cfg['activity_regularizer'],
                                    name="cnn")
        self.expand_dims = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-2))
        self.concat = keras.layers.Concatenate(axis=-2)
        self.tcn = TemporalConvNet(
            tcn_channels=self.tcn_cfg['channels'],
            groups=self.tcn_cfg['groups'],
            drop_rate=self.tcn_cfg['drop_rate'],
            kernel_regularizer=self.tcn_cfg['kernel_regularizer'],
            activity_regularizer=self.tcn_cfg['activity_regularizer'],
            name="tcn")
        self.tcn_reduced = TemporalConvNet(
            tcn_channels=self.tcn_cfg['channels'],
            groups=self.tcn_cfg['groups'],
            drop_rate=self.tcn_cfg['drop_rate'],
            kernel_regularizer=self.tcn_cfg['kernel_regularizer'],
            activity_regularizer=self.tcn_cfg['activity_regularizer'],
            padding=False,
            name="tcn_reduced")
        self.classifier = Classifier(
            units=self.class_cfg['units'],
            num_classes=self.num_classes,
            drop_rate=self.class_cfg['drop_rate'],
            kernel_regularizer=self.class_cfg['kernel_regularizer'],
            activity_regularizer=self.class_cfg['activity_regularizer'],
            name="classifier")

    def call(self, inputs, training=None):
        features = [self.cnn(inputs[:,i]) for i in range(self.frame_num)]
        features_exp = [self.expand_dims(f) for f in features]

        tcn_inp = self.concat(features_exp)
        tcn_out = self.tcn(tcn_inp)
        tcn_red_out = self.tcn_reduced(tcn_inp)

        if self.return_sequence:
            time_samples = [tcn_out[:,i] for i in range(self.frame_num)]
            x = [self.classifier(i) for i in time_samples]
            x = [self.expand_dims(i) for i in x]
            x = self.concat(x)
        else:
            x = self.classifier(tcn_out[:,self.frame_num-1])

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'cnn_cfg': self.cnn_cfg,
            'tcn_cfg': self.tcn_cfg,
            'class_cfg': self.class_cfg,

            'num_classes': self.num_classes,
            'return_sequence': self.return_sequence,
        })

        return config

    @classmethod
    def from_config(cls, config):
        obj = cls(**config)
        return obj


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class Classifier(keras.Model):
    def __init__(self, units:tuple,
                 num_classes:int,
                 drop_rate:float=0.0,
                 kernel_regularizer:float=0.0,
                 activity_regularizer:float=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.ker_reg = kernel_regularizer
        self.act_reg = activity_regularizer

        self.kernel_l2 = tf.keras.regularizers.l2(l2=kernel_regularizer)
        self.activity_l2 = tf.keras.regularizers.l2(l2=activity_regularizer)

    def build(self, input_shape):
        # print(f"Classifier build with input_shape {input_shape}")
        self.hidden = []
        for u in self.units:
            self.hidden.append(
                keras.layers.Dense(u, activation='relu',
                                    kernel_regularizer=self.kernel_l2,
                                    activity_regularizer=self.activity_l2)
            )
            self.hidden.append(
                keras.layers.Dropout(self.drop_rate)
            )
        self.dense_out = keras.layers.Dense(self.num_classes, activation='softmax',
                            kernel_regularizer=self.kernel_l2,
                            activity_regularizer=self.activity_l2)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        x = self.dense_out(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_classes': self.num_classes,
            'drop_rate': self.drop_rate,
            'kernel_regularizer': self.ker_reg,
            'activity_regularizer': self.act_reg,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class FeatureExtractor(keras.Model):
    def __init__(self,
                 channels:tuple,
                 groups:int=1,
                 drop_rate:float=0.0,
                 kernel_regularizer:float=0.0,
                 activity_regularizer:float=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.groups = groups
        self.drop_rate = drop_rate
        self.act_reg = activity_regularizer
        self.ker_reg = kernel_regularizer

        self.kernel_l2 = keras.regularizers.l2(l2=kernel_regularizer)
        self.activity_l2 = keras.regularizers.l2(l2=activity_regularizer)

    def build(self, input_shape):
        # print(f"CNN build with input shape:{input_shape}")
        self.cnn1 = keras.layers.Conv2D(self.channels[0],(5,5), activation='relu',
                                        kernel_regularizer=self.kernel_l2,
                                        activity_regularizer=self.activity_l2
                                        )
        # self.pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))
        self.bn1 = keras.layers.BatchNormalization()
        self.drop1 = keras.layers.Dropout(self.drop_rate)

        self.cnn2 = keras.layers.Conv2D(self.channels[1],(3,3), groups=self.groups, activation='relu',
                                        kernel_regularizer=self.kernel_l2,
                                        activity_regularizer=self.activity_l2
                                        )
        self.drop2 = keras.layers.Dropout(self.drop_rate)
        self.pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))
        self.bn2 = keras.layers.BatchNormalization()

        self.cnn3 = keras.layers.Conv2D(self.channels[2],(3,3), groups=self.groups, activation='relu',
                                        kernel_regularizer=self.kernel_l2,
                                        activity_regularizer=self.activity_l2
                                        )
        self.flat = keras.layers.Flatten()

    def call(self, inputs, training=None):
        x = self.cnn1(inputs)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.cnn3(x)
        x = self.flat(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'drop_rate': self.drop_rate,
            'kernel_regularizer': self.ker_reg,
            'activity_regularizer': self.act_reg,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class TemporalBlock(keras.layers.Layer):
    def __init__(self,
                 channels:int, kernel_size:int, dilation_rate:int,
                 groups:int=1,
                 drop_rate:float=0.0,
                 kernel_regularizer:float=0.0,
                 activity_regularizer:float=0.0,
                 padding:bool=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.dilation_rate = dilation_rate
        self.padding = (self.kernel_size - 1) * self.dilation_rate
        self.drop_rate = drop_rate
        self.ker_reg = kernel_regularizer
        self.act_reg = activity_regularizer
        self.pad = padding

        self.kernel_l2 = tf.keras.regularizers.l2(l2=kernel_regularizer)
        self.activity_l2 = tf.keras.regularizers.l2(l2=activity_regularizer)

    def build(self, input_shape):
        # print(f"TB build with input_shape {input_shape}")
        self.conv1 = keras.layers.Conv1D(self.channels, self.kernel_size, groups=self.groups, dilation_rate=self.dilation_rate, activation='relu',
                                         kernel_regularizer=self.kernel_l2, activity_regularizer=self.activity_l2
                                         )
        # self.conv2 = keras.layers.Conv1D(self.channels, self.kernel_size, groups=self.groups, dilation_rate=self.dilation_rate, activation='relu', kernel_regularizer=l2)
        self.add = keras.layers.Add()
        self.drop = keras.layers.Dropout(self.drop_rate)
        if self.pad:
            self.pad = keras.layers.ZeroPadding1D(padding=(self.padding, 0))
        else:
            self.crop = keras.layers.Cropping1D(cropping=(self.padding, 0))

    def call(self, inputs, training=None):
        if self.pad:
            conv_inp = self.pad(inputs)
            add_inp = inputs
        else:
            conv_inp = inputs
            add_inp = self.crop(inputs)
        x = self.conv1(conv_inp)
        x = self.add([add_inp, x])
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'groups': self.groups,
            'dilation_rate': self.dilation_rate,
            'drop_rate': self.drop_rate,
            'kernel_regularizer': self.ker_reg,
            'activity_regularizer': self.act_reg,
            'padding': self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class TemporalConvNet(keras.Model):
    def __init__(self,
                 tcn_channels:int,
                 groups:int=1,
                 drop_rate:float=0.0,
                 kernel_regularizer:float=0.0,
                 activity_regularizer:float=0.0,
                 padding:bool=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = tcn_channels
        self.groups = groups
        self.drop_rate = drop_rate
        self.act_reg = activity_regularizer
        self.ker_reg = kernel_regularizer
        self.padding = padding

        self.kernel_l2 = tf.keras.regularizers.l2(l2=kernel_regularizer)
        self.activity_l2 = tf.keras.regularizers.l2(l2=activity_regularizer)

    def build(self, input_shape):
        # print(f"TCN build with input_shape {input_shape}")
        self.use_adapt = input_shape[-1] != self.channels
        self.adapt = keras.layers.Conv1D(self.channels, kernel_size=1,
                                         kernel_regularizer=self.kernel_l2,
                                         activity_regularizer=self.activity_l2)

        self.drop = keras.layers.Dropout(self.drop_rate)
        self.block1 = TemporalBlock(self.channels, groups=self.groups, kernel_size=2, dilation_rate=1,
                                    kernel_regularizer=self.ker_reg,
                                    activity_regularizer=self.act_reg, padding=self.padding)
        self.block2 = TemporalBlock(self.channels, groups=self.groups, kernel_size=2, dilation_rate=2,
                                    kernel_regularizer=self.ker_reg,
                                    activity_regularizer=self.act_reg, padding=self.padding)
        self.block3 = TemporalBlock(self.channels, groups=self.groups, kernel_size=2, dilation_rate=4,
                                    kernel_regularizer=self.ker_reg,
                                    activity_regularizer=self.act_reg, padding=self.padding)
        self.block4 = TemporalBlock(self.channels, groups=self.groups, kernel_size=2, dilation_rate=8,
                                    kernel_regularizer=self.ker_reg,
                                    activity_regularizer=self.act_reg, padding=self.padding)

    def call(self, inputs, training=None):
        inp = inputs
        if self.use_adapt:
            inp = self.adapt(inputs)
            inp = self.drop(inp)
        x = self.block1(inp)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'tcn_channels': self.channels,
            'groups': self.groups,
            'drop_rate': self.drop_rate,
            'activity_regularizer': self.act_reg,
            'kernel_regularizer': self.ker_reg,
            'padding': self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class TimeSensitiveCCE(keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name=None):
        super().__init__(name=name)
        self.cce = keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        loss = self.cce(y_true, y_pred)
        num_frames = loss.shape[1]
        tanh = np.tanh(np.linspace(0,np.pi,num_frames))[np.newaxis,:]
        loss = tf.math.multiply(loss, tanh)
        loss = tf.math.reduce_sum(loss, axis=-1)    # reduce the time dimension with Add

        return loss

    def get_config(self):
        config = super().get_config()
        return config



# CONVOLUTIONAL NEURAL NETWORK
def get_cnn(frame_shape:tuple,
            channels:tuple,
            groups:int=1,
            drop_rate:float=0.0,
            kernel_regularizer:float=0.0,
            activity_regularizer:float=0.0) -> keras.Model:

    ker_reg = tf.keras.regularizers.l2(l2=kernel_regularizer)
    act_reg = tf.keras.regularizers.l2(l2=activity_regularizer)

    cnn_inp = keras.layers.Input(shape=frame_shape)
    cnn_x = tf.keras.layers.Conv2D(channels[0],(5,5),
        kernel_regularizer=ker_reg,
        activity_regularizer=act_reg,
        activation='relu')(cnn_inp)
    cnn_x = tf.keras.layers.Dropout(drop_rate)(cnn_x)

    cnn_x = tf.keras.layers.Conv2D(channels[1],(3,3),
        groups=groups,
        kernel_regularizer=ker_reg,
        activity_regularizer=act_reg,
        activation='relu')(cnn_x)
    cnn_x = tf.keras.layers.Dropout(drop_rate)(cnn_x)
    cnn_x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(cnn_x)

    cnn_x = tf.keras.layers.Conv2D(channels[2],(3,3),
        groups=groups,
        kernel_regularizer=ker_reg,
        activity_regularizer=act_reg,
        activation='relu')(cnn_x)
    cnn_out = tf.keras.layers.Flatten()(cnn_x)

    return keras.Model(inputs=cnn_inp, outputs=cnn_out, name='cnn')


# TEMPORAL CONVOLUTIONAL NETWORK
def get_tcn(inp_shape:tuple,
            tcn_channels:int,
            groups:int=1,
            drop_rate:float=0.0,
            kernel_regularizer:float=0.0,
            activity_regularizer:float=0.0,) -> keras.Model:

    ker_reg = tf.keras.regularizers.l2(l2=kernel_regularizer)
    act_reg = tf.keras.regularizers.l2(l2=activity_regularizer)

    tcn_inp = keras.layers.Input(shape=inp_shape)

    if(inp_shape[-1] != tcn_channels):
        temp_block_input = tf.keras.layers.Conv1D(tcn_channels, 1, groups=groups, padding='valid',
            name='adapt_conv', kernel_regularizer=ker_reg, activity_regularizer=act_reg)(tcn_inp)
        temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 1
    k=2; d=1; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels, k, groups=groups, padding='causal',
        dilation_rate=d, name='conv1', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([temp_block_input, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 2
    k=2; d=2; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels, k, groups=groups, padding='causal',
        dilation_rate=d, name='conv2', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([temp_block_input, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 3
    k=2; d=4; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels, k, groups=groups, padding='causal',
        dilation_rate=d,name='conv3', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([temp_block_input, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 4
    k=2; d=8; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels, k, groups=groups, padding='causal',
        data_format='channels_last', dilation_rate=d, activation='relu',
        name='conv4', kernel_regularizer=ker_reg)(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([temp_block_input, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    return keras.Model(inputs=tcn_inp, outputs=temp_block_input, name='tcn')


def get_tcn_reduced(inp_shape:tuple,
            tcn_channels:int,
            groups:int=1,
            drop_rate:float=0.0,
            kernel_regularizer:float=0.0,
            activity_regularizer:float=0.0,) -> keras.Model:

    ker_reg = tf.keras.regularizers.l2(l2=kernel_regularizer)
    act_reg = tf.keras.regularizers.l2(l2=activity_regularizer)

    tcn_inp = keras.layers.Input(shape=inp_shape)

    if(inp_shape[-1] != tcn_channels):
        temp_block_input = tf.keras.layers.Conv1D(tcn_channels, 1, groups=groups, padding='valid',
            name='adapt_conv', kernel_regularizer=ker_reg, activity_regularizer=act_reg)(tcn_inp)
        temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 1
    k=2; d=1; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels,k, groups=groups, padding='valid',
        dilation_rate=d, name='conv1', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    prev_crop = tf.keras.layers.Cropping1D(cropping=(padding,0))(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([prev_crop, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 2
    k=2; d=2; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels,k, groups=groups, padding='valid',
        dilation_rate=d, name='conv2', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    prev_crop = tf.keras.layers.Cropping1D(cropping=(padding,0))(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([prev_crop, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 3
    k=2; d=4; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels,k, groups=groups, padding='valid',
        dilation_rate=d,name='conv3', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    prev_crop = tf.keras.layers.Cropping1D(cropping=(padding,0))(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([prev_crop, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    # TEMPORAL BLOCK 4
    k=2; d=8; padding = (k - 1) * d
    out = tf.keras.layers.Conv1D(tcn_channels,k, groups=groups, padding='valid',
        dilation_rate=d, name='conv4', activation='relu',
        kernel_regularizer=ker_reg, activity_regularizer=act_reg)(temp_block_input)
    prev_crop = tf.keras.layers.Cropping1D(cropping=(padding,0))(temp_block_input)
    temp_block_input = tf.keras.layers.Add()([prev_crop, out])
    temp_block_input = tf.keras.layers.Dropout(drop_rate)(temp_block_input)

    tcn_reduced = tf.keras.Model(tcn_inp, temp_block_input, name='tcn_reduced')
    return tcn_reduced

def get_classifier(input_shape:tuple,
                num_classes:int,
                units:tuple,
                drop_rate:float=0.0,
                kernel_regularizer:float=0.0,
                activity_regularizer:float=0.0,) -> keras.Model:

    ker_reg = tf.keras.regularizers.l2(l2=kernel_regularizer)
    act_reg = tf.keras.regularizers.l2(l2=activity_regularizer)
    out = class_inp = tf.keras.Input(input_shape, name='classifier_in')

    for u in units:
        out = tf.keras.layers.Dense(u, activation='relu', kernel_regularizer=ker_reg, activity_regularizer=act_reg)(out)
        out = tf.keras.layers.Dropout(drop_rate)(out)
    out = tf.keras.layers.Dense(num_classes, kernel_regularizer=ker_reg, activity_regularizer=act_reg)(out)
    out = tf.keras.layers.Softmax()(out)

    classifier = tf.keras.Model(class_inp, out, name='classifier')
    return classifier

def get_tinyssimonn(input_shape:tuple,
                    cnn_cfg:dict,
                    tcn_cfg:dict,
                    class_cfg:dict,
                    num_classes:int) -> keras.Model:

    inp = tf.keras.Input(input_shape, name='tinyssimonn_in')
    cnn_input_shape = input_shape[1:]
    num_frames = input_shape[0]


    cnn = get_cnn(cnn_input_shape, cnn_cfg['channels'], groups=cnn_cfg['groups'],
                    drop_rate=cnn_cfg['drop_rate'], kernel_regularizer=cnn_cfg['kernel_regularizer'],
                    activity_regularizer=cnn_cfg['activity_regularizer'])

    features = [cnn(inp[:,i]) for i in range(num_frames)]
    features = [keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-2))(f) for f in features]
    features = keras.layers.Concatenate(axis=-2)(features)


    tcn = get_tcn((num_frames, *cnn.output.shape[1:]), tcn_cfg['channels'], groups=tcn_cfg['groups'],
                    drop_rate=tcn_cfg['drop_rate'], kernel_regularizer=tcn_cfg['kernel_regularizer'],
                    activity_regularizer=tcn_cfg['activity_regularizer'])

    tcn_out = tcn(features)

    classifier = get_classifier((tcn.output.shape[-1],), num_classes, class_cfg['units'],
                    drop_rate=class_cfg['drop_rate'], kernel_regularizer=class_cfg['kernel_regularizer'],
                    activity_regularizer=class_cfg['activity_regularizer'])

    class_inp = [tcn_out[:,i] for i in range(num_frames)]
    class_out = [classifier(i) for i in class_inp]

    out = tf.keras.layers.Concatenate()(class_out)
    out = tf.keras.layers.Reshape((num_frames, -1))(out)

    return keras.Model(inputs=inp, outputs=out, name='full')
