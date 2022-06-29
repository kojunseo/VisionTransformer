import tensorflow as tf
from VisionTransformer.model import ViT
from VisionTransformer.params import params
from tensorflow.keras.datasets import cifar100
import tensorflow_addons as tfa


(X_train, y_train), (X_test, y_test) = cifar100.load_data()

model = ViT(params)
model.build([None, 32,32,3])
model.load_weights(params["checkpoint_path"])
model.summary()
# this_weight = None
# for weight in model.get_layer("transformer_layer_10").get_layer("multi_head_attention_9").weights:
#     print(weight.name, weight.shape)
#     if weight.name == "transformer_layer_10/multi_head_attention_9/attention_output/kernel:0":
#         this_weight = weight

# model_att = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer("transformer_layer_9").weights, model.output])
output, attention_score = model.predict(X_test[0:1])
import numpy as np
import cv2
print(np.argmax(output, axis=1))

attention_score = np.mean(attention_score, axis=1)
attention_score = np.sum(attention_score, axis=1)
norm_score = (attention_score - np.min(attention_score)) / (np.max(attention_score) - np.min(attention_score))
norm_score = norm_score.reshape(12, 12)
norm_score_resize = cv2.resize(norm_score, (32, 32)).reshape(-1, 32, 32, 1)
print(norm_score_resize.shape)

cv2.imwrite("./image.png", cv2.resize(X_test[0], (72, 72)))
cv2.imwrite("./masked_image.png", cv2.resize(X_test[0]*norm_score_resize[0], (72, 72)))

print(X_test[0].shape)