import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

# This script contains the ActorCritic class, so the network used to train the agents.
# As explained in the report, I used a single network both for policy and value due to computational and time constraints.
# An attempt to use Residual Blocks was made, but I would have not made it in time for the deadline. 

@register_keras_serializable()
class ActorCritic(tf.keras.Model):
    def __init__(self, n_actions, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.n_actions = n_actions
        self.conv_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, kernel_size=8, strides=2, activation="relu", input_shape=(64, 64, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, kernel_size=2, strides=1, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, kernel_size=2, strides=1, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
            ]
        )

        self.actor_head = tf.keras.layers.Dense(self.n_actions, activation="softmax")
        self.critic_head = tf.keras.layers.Dense(1)

    # Forward pass
    def call(self, x):
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x)

        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=0)  # Add batch dimension if missing (for batch processing)

        # Normalization to range [0,1] to stabilize training
        x = tf.cast(x, tf.float32) / 255.0
        initial_computation = self.conv_layers(x)
        probs = self.actor_head(initial_computation)
        value = self.critic_head(initial_computation)
        return probs, value
    
    # Necessary to enable serialization and desarialization of a custom Keras model
    # so that it can be saved and loaded in the right way
    def get_config(self):
        config = super().get_config()
        config.update({"n_actions": self.n_actions})
        return config