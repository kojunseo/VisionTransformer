import tensorflow as tf
from .layers import MLPLayer, AugmentationLayer, Patches, PatchEncoder, TransformerLayer

class ViT(tf.keras.Model):
    def __init__(self, params):
        super(ViT, self).__init__()
        self.patch_size = params["patch_size"]
        self.num_classes = params["num_classes"]
        self.image_size = params["image_size"]
        self.num_patches = params["num_patches"]
        self.projection_dim = params["projection_dim"]
        self.num_heads = params["num_heads"]
        self.num_mlp_head_units = params["mlp_head_units"]
        self.num_transformer_units = params["transformer_units"]
        self.num_transformer_layers = params["transformer_layers"]

        self.augmentation_layer = AugmentationLayer(self.image_size)
        self.patches = Patches(self.patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, self.projection_dim)
        self.transformer_layers = [TransformerLayer(self.num_heads, self.projection_dim, self.num_transformer_units, dropout_rate = 0.1, name=f"transformer_layer_{i+1}") for i in range(self.num_transformer_layers)]
        self.normalization_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(rate=0.3)

        self.features_layer = MLPLayer(self.num_mlp_head_units, dropout_rate = 0.3)
        self.classification_layer = tf.keras.layers.Dense(self.num_classes, activation="softmax")

    
    def call(self, inputs, training=True):
        x = self.augmentation_layer(inputs)
        x = self.patches(x)
        x = self.patch_encoder(x)
        for transformer_layer in self.transformer_layers[:-1]:
            x = transformer_layer(x)
        if training:
            x = self.transformer_layers[-1](x)
        else:
            x, att_score = self.transformer_layers[-1](x, return_attention_scores=training)
        x = self.normalization_layer(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.features_layer(x)
        x = self.classification_layer(x)
        if training:
            return x
        else:
            return x, att_score
    