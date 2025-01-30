from memory import Memory
from network import ActorCritic
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

class PPOAgent:
    def __init__(
        self,
        env,
        game,
        total_steps,
        epochs,
        batch_size,
        n_actions,
        model_path,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        learning_rate=3e-4,
    ):
        self.env = env
        self.epochs = epochs
        self.game = game
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.model_path = model_path
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.memory = Memory(batch_size)
        self.actor_critic = ActorCritic(n_actions)
        
        self.lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.total_steps,
            end_learning_rate=1e-5
        )

        self.entropy_coef_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.05, 
            decay_steps=self.total_steps,          
            end_learning_rate=0.001 
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler)

    def store(self, state, action, probs, values, reward, done):
        self.memory.store(state, action, probs, values, reward, done)

    def sample_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        probs, value = self.actor_critic(state)
        distribution = tfp.distributions.Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()[0]

        return action, log_prob, value
    
    # It's important because it helps the agent to learn better by focusing more on actions that lead to good rewards
    def compute_advantages(self, rewards, values, dones):
        steps = len(rewards)
        advantages = np.zeros(steps, dtype=np.float32)
        gae = 0
        next_value = 0

        for step in reversed(range(steps)):
            next_value = values[step + 1] if step + 1 < steps else 0
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae

        return advantages

    def compute_entropy(self, probabilities):
        # Add a small epsilon (to not confuse with the epsilon that we use to clip) to avoid log(0)
        safe_probabilities = probabilities + 1e-10
        return -tf.reduce_sum(safe_probabilities * tf.math.log(safe_probabilities), axis=-1)     
    
    def train(self):
        for _ in range(self.epochs):
            states, actions, old_probs, old_values, rewards, dones, batches = self.memory.generate_batches()
            advantages = self.compute_advantages(rewards, old_values, dones)

            for batch in batches:
                with tf.GradientTape() as tape:                    
                    states_batch = tf.convert_to_tensor(states[batch])
                    old_probs_batch = tf.convert_to_tensor(old_probs[batch])
                    actions_batch = tf.convert_to_tensor(actions[batch])

                    probs, value = self.actor_critic(states_batch)
                    value = tf.squeeze(value, 1)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions_batch)

                    prob_ratio = tf.math.exp(new_probs - old_probs_batch)
                    weighted_probs = advantages[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.epsilon, 1+self.epsilon)
                    weighted_clipped_probs = clipped_probs * advantages[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    # Entropy loss: to balance exploration and exploitation
                    # at first the agent explores more thanks to an higher entropy coefficient
                    # as the training continues the agent starts to exploit more its knowledge in order to achieve the highest cumulative reward
                    entropy_coef = self.entropy_coef_scheduler(self.optimizer.iterations)
                    entropy_loss = tf.reduce_mean(self.compute_entropy(probs))
                    actor_loss -= entropy_coef * entropy_loss 

                    returns = advantages[batch] + old_values[batch]

                    critic_loss = tf.keras.losses.MSE(value, returns)
                    ## Another way to calculate the critic loss, but I chose to use tf.keras.losses.MSE to have more optimization
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(returns-critic_value, 2))

                    total_loss = actor_loss + 0.5 * critic_loss

                    ## DEBUG
                    # print(f"Actor Loss: {actor_loss.numpy()}, Critic Loss: {critic_loss.numpy()}, Entropy Loss: {entropy_loss.numpy()}")

                # Backward pass
                # Clipped the gradient to have a stable training 
                grads, _ = tf.clip_by_global_norm(tape.gradient(total_loss, self.actor_critic.trainable_variables), 0.5)
                self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))

        self.memory.clear()

    def save_model(self, version):
        save_dir = f"{self.model_path}/{self.game}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # print("##### Saving model ...")
        self.actor_critic.save(f"{save_dir}/{version}.keras")

    def load_model(self, version):
        print("##### Loading model ...")
        self.actor_critic = tf.keras.models.load_model(f"{self.model_path}/{self.game}/{version}.keras")