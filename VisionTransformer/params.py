params = {
    "learning_rate" : 0.001,
    "weight_decay" : 0.0001,
    "batch_size" : 256,
    "num_epochs" : 100,
    "image_size" : 72, 
    "patch_size" : 6,  
    "num_patches" : (72 // 6) ** 2, # 72 = image_size, 6 = patch_size
    "projection_dim" : 64,
    "num_heads" : 5,
    "transformer_units" : [
        64 * 2, # 64 = projection_dim
        64,
    ],  
    "transformer_layers" : 10,
    "mlp_head_units" : [2048, 1024],
    "num_classes": 100,
    "input_size": (32, 32, 3),
    "checkpoint_path": './VisionTransformer/weights/model.h5',
    }