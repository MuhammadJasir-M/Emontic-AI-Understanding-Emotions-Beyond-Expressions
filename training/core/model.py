# training/core/model.py
# Production-Tier EfficientNetV2-S with Position-Aware Multi-Head Self-Attention.
# Upgraded to 224x224 input grid density with localized Attention Pooling.

import tensorflow as tf
import keras
from keras import layers

@keras.saving.register_keras_serializable(package="Custom")
class SpatialPositionalEmbedding(layers.Layer):
    """
    Learnable 2D positional embedding block broadcasted across any batch size.
    Dynamically sizes its coordinate matrix based on the input sequence length,
    ensuring compliance with both 112x112 and 224x224 grid configurations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_embedding = None
        
    def build(self, input_shape):
        # Dynamically map token dimensions based on incoming feature maps
        num_tokens = input_shape[1]  # 16 for 112x112 input, 49 for 224x224 input
        channels = input_shape[2]    # 1280 for EfficientNetV2-S
        
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, num_tokens, channels),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        return inputs + self.pos_embedding
        
    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="Custom")
class AttentionPooling(layers.Layer):
    """
    Multi-Head Contextual token aggregation layer.
    Uses multiple learnable query vectors to capture diverse facial action units 
    (e.g., eyes, mouth) simultaneously, projecting them back into a dense representation.
    """
    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.query_matrix = None
        self.projection = None

    def build(self, input_shape):
        channels = input_shape[-1]
        self.query_matrix = self.add_weight(
            name="query_matrix",
            shape=(1, self.num_heads, channels),
            initializer="glorot_uniform",
            trainable=True
        )
        self.projection = layers.Dense(channels, activation="swish")
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, tokens, channels)
        # scores: (batch, tokens, num_heads)
        scores = tf.matmul(inputs, self.query_matrix, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=1)
        
        # weights: (batch, tokens, num_heads, 1)
        w_expanded = tf.expand_dims(weights, -1)
        # inputs: (batch, tokens, 1, channels)
        i_expanded = tf.expand_dims(inputs, 2)
        
        # pooled: (batch, num_heads, channels)
        pooled = tf.reduce_sum(i_expanded * w_expanded, axis=1)
        
        # flatten: (batch, num_heads * channels)
        channels = inputs.shape[-1]
        flattened = tf.reshape(pooled, [-1, self.num_heads * channels])
        
        return self.projection(flattened)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config


@keras.saving.register_keras_serializable(package="Custom")
class ArcFaceDense(layers.Layer):
    """
    Final Dense Layer for ArcFace.
    Normalizes both the input features and the learned weights onto a hypersphere,
    returning the scaled cosine similarity (logits).
    """
    def __init__(self, num_classes, scale=30.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights",
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # L2 Normalize inputs and weights
        x = tf.math.l2_normalize(inputs, axis=1)
        w = tf.math.l2_normalize(self.w, axis=0)
        
        # Calculate Cosine Similarity
        cos_theta = tf.matmul(x, w)
        
        # Scale to return as logits
        return cos_theta * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "scale": self.scale})
        return config


def build_production_model(input_shape=(224, 224, 3), num_classes=7):
    """
    Production-Tier EfficientNetV2-S with Position-Aware Multi-Head Self-Attention.
    Upgraded to capture micro-expressions via increased input resolution and attention pooling.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # 1. Base Backbone Feature Extractor
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet"
    )
    base_model._name = "efficientnetv2-s"
    base_model.trainable = False  # Controlled dynamically via optimization schedules
    
    # At 224x224 input, output features shape is (None, 7, 7, 1280)
    features = base_model(inputs)  
    
    # 2. Spatial Tokenization Block (7x7 grid collapses to 49 distinct facial tokens)
    tokens = layers.Reshape((-1, 1280), name="spatial_tokenizer")(features)
    
    # 3. Inject Spatial Position Coordinates
    encoded_tokens = SpatialPositionalEmbedding(name="positional_injection")(tokens)
    
    # 4. Multi-Head Self-Attention (8 Heads, 128 Key Dimension)
    attention_out = layers.MultiHeadAttention(
        num_heads=8, 
        key_dim=128, 
        name="spatial_self_attention"
    )(query=encoded_tokens, value=encoded_tokens)
    
    # Residual Connection + Layer Normalization
    attended_context = layers.Add(name="attention_residual")([encoded_tokens, attention_out])
    attended_context = layers.LayerNormalization(name="attention_norm")(attended_context)
    
    # 5. Contextual Sequence Pooling (Replaces unweighted GAP_1D)
    x = AttentionPooling(name="contextual_attention_pooling")(attended_context)
    
    # 6. Production Dense Classification Head Hierarchy
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)
    
    x = layers.Dense(512, activation="swish", name="fc1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    
    x = layers.Dense(256, activation="swish", name="fc2")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)
    
    # ArcFace Margin Prediction Head (Outputs scaled logits instead of probabilities)
    outputs = ArcFaceDense(num_classes, scale=30.0, name="predictions")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="umontic_production_v2s")


def get_model(arch: str, trainable_base: bool = False, num_classes: int = 7):
    """
    Pipeline Factory Entry Point for model instantiation.
    """
    if arch.lower() != "v2s":
        raise ValueError(f"Unsupported architecture backbone configuration: {arch}")
        
    model = build_production_model(num_classes=num_classes)
    
    if trainable_base:
        # Default fallback initialization unfreezes baseline layers
        unfreeze_top_layers(model, num_layers=120)
        
    return model


def unfreeze_top_layers(model, num_layers: int):
    """
    Surgically unfreezes backbone parameters by functional block names rather than flat indexes.
    Protects Batch Normalization layers to avoid destroying moving statistics.
    Backward-compatible parameter signature maps to stage boundaries.
    """
    # Outer head configurations are always trainable
    for layer in model.layers:
        if layer.name != "efficientnetv2-s":
            layer.trainable = True

    try:
        backbone = model.get_layer("efficientnetv2-s")
        backbone.trainable = True
        
        # Map raw layer counts to explicit block structures to ensure structural integrity
        if num_layers <= 60:
            target_blocks = ["block6"]  # Stage 1 Pretraining strategy
        else:
            target_blocks = ["block5", "block6"]  # Stage 2 Fine-tuning strategy
            
        unfrozen_count = 0
        for layer in backbone.layers:
            # Strictly freeze Batch Normalization across the entire backbone graph
            if "bn" in layer.name or "normalization" in layer.name.lower():
                layer.trainable = False
            # Unfreeze only if the layer belongs to a targeted block group
            elif any(block in layer.name for block in target_blocks):
                layer.trainable = True
                unfrozen_count += 1
            else:
                layer.trainable = False
                
        print(f"🔓 [BLOCK-AWARE UNFREEZE] Enabled parameters for {unfrozen_count} layers matching {target_blocks} structures.")
        
    except ValueError:
        print("[WARNING] 'efficientnetv2-s' layer container missing. Falling back to index cutoff mapping.")
        cutoff = len(model.layers) - num_layers
        for i, layer in enumerate(model.layers):
            if "bn" in layer.name or "normalization" in layer.name.lower():
                layer.trainable = False
            else:
                layer.trainable = (i >= cutoff)