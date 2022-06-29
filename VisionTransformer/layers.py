import tensorflow as tf

class mlpBlock(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate):
        super(mlpBlock, self).__init__()
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense_1 = tf.keras.layers.Dense(units=units, activation=tf.nn.gelu)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        x = self.dropout(x, training=training)
        return x

class MLPLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout_rate):
        super(MLPLayer, self).__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.mlp_blocks = [mlpBlock(units=hidden_units[i], dropout_rate=dropout_rate) for i in range(len(hidden_units))]

    def call(self, x, training=None):
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x, training)
        return x

class AugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, image_size):
        super(AugmentationLayer, self).__init__()
        self.image_size = image_size
        self.normalization = tf.keras.layers.Normalization()
        self.resizing = tf.keras.layers.Resizing(height = image_size, width = image_size)
        self.random_flip = tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal")
        self.random_rotation = tf.keras.layers.experimental.preprocessing.RandomRotation(
            factor=0.07)
        self.random_zoom = tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=0.3, width_factor=0.3)

    def call(self, inputs):
        x = self.normalization(inputs)
        x = self.resizing(x)
        x = self.random_flip(x)
        x = self.random_rotation(x)
        x = self.random_zoom(x)
        return x

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        print("input images: ", images.shape)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        print("patches: ", patches.shape)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        print("reshape patches: ", patches.shape)
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        print("patch input: ", patch.shape)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        print("position: ", positions.shape)
        encoded = self.projection(patch) + self.position_embedding(positions)
        print("embedding: ", self.position_embedding(positions).shape)
        print("projection: ", self.projection(patch).shape)
        return encoded

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, projection_dim, transformer_units, dropout_rate, name):
        super(TransformerLayer, self).__init__(name=name)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.transformer_units = transformer_units
        self.dropout_rate = dropout_rate

        self.layer_normalization1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normalization2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim = self.projection_dim,
            dropout = self.dropout_rate,
        )
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()
        self.mlp = MLPLayer(hidden_units = self.transformer_units, dropout_rate = self.dropout_rate)

    def call(self, inputs, return_attention_scores=False):
        x = self.layer_normalization1(inputs)
        if return_attention_scores:
            x, attention_score = self.multi_head_attention(x, x, return_attention_scores=return_attention_scores)
        else:
            x = self.multi_head_attention(x, x)

        x1 = self.add1([x, inputs])
        x2 = self.layer_normalization2(x1)
        x2 = self.mlp(x2)
        x = self.add2([x1, x2])
        
        if return_attention_scores:
            return x, attention_score
        else:
            return x

    