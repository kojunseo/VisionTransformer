import numpy as np
from tensorflow.keras.datasets import cifar100
import tensorflow as tf
import tensorflow_addons as tfa

# cifar 데이터 불러오기
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

from ResidualAttentionNetwork.model import ResidualAttentionNetwork
from VisionTransformer.model import ViT
from VisionTransformer.params import params


model = ViT(params)
optimizer = tfa.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.0001
    )
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=params["checkpoint_path"],
                                        save_weights_only=True,
                                        save_best_only=True,
                                        verbose=1)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)
model.fit(X_train, y_train, batch_size = 256, epochs=250, validation_split= 0.1, callbacks=[checkpoint_callback])
model.load_weights('./VisionTransformer/weights/model.h5')

_, acc, top5_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', acc)
print('Test top-5 accuracy:', top5_acc)