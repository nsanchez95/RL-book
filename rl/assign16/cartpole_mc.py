### USED FOR HELP IN ADAPTING TENSORFLOW CODE:
### https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic


import numpy as np
import tensorflow as tf 
import gym
import tensorflow_probability as tfp
from typing import Sequence, Optional, Tuple, List
from dataclasses import dataclass
import tqdm

# env= gym.make("LunarLander-v2")
env = gym.make("CartPole-v0")
low = env.observation_space.low
high = env.observation_space.high

@dataclass(frozen=True)
class DNNSpec_TF:
    neurons: Sequence[int]
    activations: Sequence[Optional[str]]




### wrapping env call - taken from https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])


eps = eps = np.finfo(np.float32).eps.item()
def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns

#####################################
# gets a full episode rewards and action probabilities for fixed policy
def get_episode(
    start_state: tf.Tensor, 
    policy_approx: tf.keras.Model,
    lim_steps: int):
    action_logprobs = tf.TensorArray(dtype =tf.float32, size = 0, dynamic_size = True)
    rewards = tf.TensorArray(dtype =tf.int32, size = 0, dynamic_size = True)

    state = start_state
    for t in tf.range(lim_steps):
        logits_actions = policy_approx(tf.expand_dims(state,axis = 0))
        action_sample = tf.random.categorical(logits = logits_actions,num_samples = 1)[0,0]
        action_logprobs_t = tf.nn.log_softmax(logits_actions)
        state, reward, done = tf_env_step(action_sample)

        # store results for later
        action_logprobs = action_logprobs.write(t, action_logprobs_t[0, action_sample])
        rewards = rewards.write(t, reward)
        if tf.cast(done, tf.bool):
            break
    rewards = rewards.stack()
    action_logprobs = action_logprobs.stack()
    return rewards, action_logprobs


def compute_loss(action_logprobs:tf.Tensor,returns:tf.Tensor) -> tf.Tensor:
    return -tf.math.reduce_sum(action_logprobs*tf.stop_gradient(returns))

def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    rewards, action_logprobs = get_episode(
        initial_state, model, max_steps_per_episode) 
    returns = get_expected_return(rewards, gamma)
    action_logprobs, returns = [
        tf.expand_dims(x, 1) for x in [action_logprobs, returns]] 
    loss = compute_loss(action_logprobs, returns)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  episode_reward = tf.math.reduce_sum(rewards)
  return episode_reward

def create_nn(dnn_spec : DNNSpec_TF):
    return tf.keras.models.Sequential([tf.keras.layers.Dense(num_ins,activation=act_fn)
                for num_ins,act_fn in zip(dnn_spec.neurons, dnn_spec.activations)])




#################### ACTUAL LOOP ###################
agent_spec_nn = DNNSpec_TF(
    neurons = [128, env.action_space.n],
    activations = ['relu', None])

agent = create_nn(agent_spec_nn)
agent_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99
rew_hist = []
with tqdm.trange(max_episodes) as t:
  for i in t:
    initial_state = tf.constant(env.reset(), dtype=tf.float32)
    episode_reward = int(train_step(
        initial_state, agent, agent_optimizer, gamma, max_steps_per_episode))

    running_reward = episode_reward*0.01 + running_reward*.99

    t.set_description(f'Episode {i}')
    t.set_postfix(running_reward=running_reward,
        episode_reward=episode_reward)
    rew_hist.append(episode_reward)
    # Show average episode reward every 10 episodes
    if i % 100 == 0:
      np.savetxt('save_eps_mc_lambda_2n5',np.array(rew_hist))

    if running_reward > reward_threshold:  
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
