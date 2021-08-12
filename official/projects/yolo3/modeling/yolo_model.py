## modifed from exmaple_model.py => yolo3_model.py

"""A sample model implementation.
This is only a dummy example to showcase how a model is composed. It is usually
not needed to implement a modedl from scratch. Most SoTA models can be found and
directly used from `official/vision/beta/modeling` directory.
"""

import tensorflow as tf
# from official.vision.beta.projects.yolo3 import yolo3_config 


@tf.keras.utils.register_keras_serializable(package='Vision')
class Yolo3Model(tf.keras.Model):
    """A example model class.
    A model is a subclass of tf.keras.Model where layers are built in the
    constructor.
    """
    def __init__(
        self,
        num_classes: int,
        input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
            shape=[None, None, None, 3]),
        **kwargs):
        """Initializes the example model.
        All layers are defined in the constructor, and config is recorded in the
        `_config_dict` object for serialization.
        Args:
        num_classes: The number of classes in classification task.
        input_specs: A `tf.keras.layers.InputSpec` spec of the input tensor.
        **kwargs: Additional keyword arguments to be passed.
        """
        inputs = tf.keras.Input(shape=input_specs.shape[1:], name=input_specs.name)

        # Layer 0 => 1 : 
        outputs = self.conv_block(inputs, [
            {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
            {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1}
        ], skip=False)

        # Layer 2 => 4 : Layer 4 Residual
        outputs = self.conv_block(outputs, [    
            {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
            {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}
        ], skip=True)

        # Layer 5 
        outputs = self.conv_block(outputs, [
            {'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5}
        ], skip=False)

        # Layer 6 => 11 : Layer 8/11 Residual
        for i in range(2):
            outputs = self.conv_block(outputs, [
                {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6 + i * 3},
                {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7 + i * 3}
            ], skip=True)

        # Layer 12 
        outputs = self.conv_block(outputs, [
            {'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12}
        ], skip=False)

        # Layer 13 => 36
        for i in range(8):
            outputs = self.conv_block(outputs, [
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13 + i * 3},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14 + i * 3}
            ], skip=True)
        skip_36 = outputs

        # Layer 37 => 40
        outputs = self.conv_block(outputs, [
            {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37}
        ], skip=False)

        # Layer 41 => 61
        for i in range(8):
            outputs = self.conv_block(outputs, [
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38 + i * 3},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39 + i * 3}
            ])
        skip_61 = outputs

        # Layer 62 
        outputs = self.conv_block(outputs, [
            {'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62}
        ], skip=False)

        # Layer 63 => 74
        for i in range(4):
            outputs = self.conv_block(outputs, [
                {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63 + i * 3},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64 + i * 3}
            ], skip=True)

        # Layer 75 => 80
        for i in range(3):
            outputs = self.conv_block(outputs, [
                {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75 + i * 2},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76 + i * 2}
            ], skip=False)

        ## TODO: check the Layer 81

        # Layer 82
        yolo_82 = self.conv_block(outputs, [
            {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}
        ], skip=False)

        ## TODO: check layer_idx
        # Layer 83 => 86
        outputs = self.conv_block(outputs, [
            {'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}, 
        ], skip=False)
        outputs = tf.keras.layers.UpSampling2D(2)(outputs)
        outputs = tf.keras.layers.concatenate([outputs, skip_61])

        # Layer 87 => 92
        for i in range(3):
            outputs = self.conv_block(outputs, [
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87 + i * 2},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88 + i * 2}
            ], skip=False)

        # Layer 93 => 94
        yolo_94 = self.conv_block(outputs, [
            {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}
        ])

        # Layer 95 => 98
        outputs = self.conv_block(outputs, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96},
        ])
        outputs = tf.keras.layers.UpSampling2D(2)(outputs)
        outputs = tf.keras.layers.concatenate([outputs, skip_36])

        # Layer 99 => 106
        yolo_106 = self.conv_block(outputs, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 100},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 101},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 102},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 103},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 104},
            {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}
        ])
        
        # final model 
        self.model = tf.keras.Model(inputs, [yolo_82, yolo_94, yolo_106], name='yolo3_model')

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self._input_specs = input_specs
        self._config_dict = {
            'num_classes': num_classes, 
            'input_specs': input_specs
            }


    def conv_block(self, input_img, convs, skip=True):
        x = input_img
        count = 0
        for conv in convs:
            if count == (len(convs) - 2) and skip:
                skip_connection = x
            count += 1
            if conv['stride'] > 1:
                # peculiar padding as darknet prefer left and top
                x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            x = tf.keras.layers.Conv2D(
                    filters=conv['filter'],
                    kernel_size=conv['kernel'],
                    strides=conv['stride'],
                    # peculiar padding as darknet prefer left and top
                    padding='valid' if conv['stride'] > 1 else 'same',
                    name='conv_' + str(conv['layer_idx']), 
                    use_bias=False if conv['bnorm'] else True )(x)
            if conv['bnorm']:
                x = tf.keras.layers.BatchNormalization(
                    name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']:
                x = tf.keras.layers.LeakyReLU(
                    alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
        return skip_connection + x if skip else x

