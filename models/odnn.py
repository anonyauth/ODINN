import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import layers
from models.base_odnn import BaseODINN

class ODINN(BaseODINN):
    def inference(inputs, nb_classes, training, ffd_drop,
            norm_mat, output_dim, activation=tf.nn.elu):
        
        h_1 = layers.DeGroot(inputs, norm_mat=norm_mat,
            output_dim=output_dim, activation=activation,
            in_drop=ffd_drop)

        # h_1 = layers.Friedkin_Johnsen(inputs, norm_mat=norm_mat,
        #     output_dim=output_dim, activation=activation,
        #     in_drop=ffd_drop)
        
        logits = layers.linear_layer(h_1,
            output_dim=nb_classes, activation=lambda x: x,
            in_drop=ffd_drop)
    
        return logits
