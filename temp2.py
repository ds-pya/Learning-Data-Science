import tensorflow as tf
from tensorflow.keras import layers, Model

def TabTransformer(user_feature_dim, item_feature_dim, num_categories, embed_dim, num_heads, ff_dim, num_transformer_blocks):
    # Embedding layers for user and item features
    user_input = layers.Input(shape=(user_feature_dim,), name="user_feature")
    item_input = layers.Input(shape=(item_feature_dim,), name="item_feature")

    # Embedding for categorical features
    user_embeddings = layers.Embedding(input_dim=num_categories, output_dim=embed_dim)(user_input)
    item_embeddings = layers.Embedding(input_dim=num_categories, output_dim=embed_dim)(item_input)

    # Concatenate embeddings
    x = layers.Concatenate(axis=1)([user_embeddings, item_embeddings])
    
    # Positional Encoding
    x = layers.LayerNormalization()(x)
    
    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        attention_output = layers.Dropout(0.1)(attention_output)
        attention_output = layers.Add()([x, attention_output])  # Residual Connection
        attention_output = layers.LayerNormalization()(attention_output)

        # Feed-Forward Network
        ffn_output = layers.Dense(ff_dim, activation="relu")(attention_output)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.Add()([attention_output, ffn_output])  # Residual Connection
        x = layers.LayerNormalization()(x)

    # Pooling layer to summarize sequence
    x = layers.GlobalAveragePooling1D()(x)
    
    # Fully Connected Layer
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="linear", name="output")(x)

    # Create Model
    model = Model(inputs=[user_input, item_input], outputs=output)
    return model

# Model Parameters
user_feature_dim = 10  # e.g., user categorical feature dimensions
item_feature_dim = 15  # e.g., item categorical feature dimensions
num_categories = 100   # maximum category value in any feature
embed_dim = 32         # embedding dimension
num_heads = 4          # number of attention heads
ff_dim = 64            # feed-forward network dimension
num_transformer_blocks = 2  # number of transformer blocks

# Build the TabTransformer Model
tab_transformer_model = TabTransformer(user_feature_dim, item_feature_dim, num_categories, embed_dim, num_heads, ff_dim, num_transformer_blocks)

# Compile the model
tab_transformer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                               loss='mse',  # Mean Squared Error for regression
                               metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Summary
tab_transformer_model.summary()


import tensorflow as tf
from tensorflow.keras import layers, Model

def CORALModel(user_feature_dim, item_feature_dim, num_categories, embed_dim, hidden_dim):
    # Inputs
    user_input = layers.Input(shape=(user_feature_dim,), name="user_feature")
    item_input = layers.Input(shape=(item_feature_dim,), name="item_feature")

    # Embedding layers
    user_embeddings = layers.Embedding(input_dim=num_categories, output_dim=embed_dim)(user_input)
    item_embeddings = layers.Embedding(input_dim=num_categories, output_dim=embed_dim)(item_input)

    # Concatenate embeddings
    x = layers.Concatenate(axis=1)([user_embeddings, item_embeddings])
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layers
    x = layers.Dense(hidden_dim, activation="relu")(x)
    x = layers.Dense(hidden_dim, activation="relu")(x)

    # Output for CORAL: k-1 outputs for binary classification
    coral_outputs = layers.Dense(7, activation="sigmoid", name="coral_output")(x)

    # Model
    model = Model(inputs=[user_input, item_input], outputs=coral_outputs)
    return model

# Parameters
user_feature_dim = 10
item_feature_dim = 15
num_categories = 100
embed_dim = 32
hidden_dim = 64

# Build CORAL Model
coral_model = CORALModel(user_feature_dim, item_feature_dim, num_categories, embed_dim, hidden_dim)

# Custom CORAL Loss
def coral_loss(y_true, y_pred):
    k = tf.shape(y_pred)[1] + 1
    y_true = tf.cast(y_true, tf.float32)
    y_true_expanded = tf.concat([tf.expand_dims(y_true, axis=-1)] * (k - 1), axis=-1)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_expanded, logits=y_pred))
    return loss

# Compile CORAL Model
coral_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=coral_loss,
                    metrics=['accuracy'])

# Summary
coral_model.summary()

def CORNModel(user_feature_dim, item_feature_dim, num_categories, embed_dim, hidden_dim):
    # Inputs
    user_input = layers.Input(shape=(user_feature_dim,), name="user_feature")
    item_input = layers.Input(shape=(item_feature_dim,), name="item_feature")

    # Embedding layers
    user_embeddings = layers.Embedding(input_dim=num_categories, output_dim=embed_dim)(user_input)
    item_embeddings = layers.Embedding(input_dim=num_categories, output_dim=embed_dim)(item_input)

    # Concatenate embeddings
    x = layers.Concatenate(axis=1)([user_embeddings, item_embeddings])
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layers
    x = layers.Dense(hidden_dim, activation="relu")(x)
    x = layers.Dense(hidden_dim, activation="relu")(x)

    # Output for CORN: k-1 outputs for regression
    corn_outputs = layers.Dense(7, activation="linear", name="corn_output")(x)

    # Model
    model = Model(inputs=[user_input, item_input], outputs=corn_outputs)
    return model

# Build CORN Model
corn_model = CORNModel(user_feature_dim, item_feature_dim, num_categories, embed_dim, hidden_dim)

# Custom CORN Loss
def corn_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.sequence_mask(y_true, maxlen=tf.shape(y_pred)[1])
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(y_pred_masked), logits=y_pred_masked))
    return loss

# Compile CORN Model
corn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss=corn_loss,
                   metrics=['mae'])

# Summary
corn_model.summary()