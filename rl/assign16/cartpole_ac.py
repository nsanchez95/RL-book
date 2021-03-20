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
    ac_model: tf.keras.Model,
    lim_steps: int):
    action_logprobs = tf.TensorArray(dtype =tf.float32, size = 0, dynamic_size = True)
    rewards = tf.TensorArray(dtype =tf.int32, size = 0, dynamic_size = True)
    critic_values = tf.TensorArray(dtype =tf.float32, size = 0, dynamic_size = True)

    state = start_state
    for t in tf.range(lim_steps):
        logits_actions, critic_value = ac_model(tf.expand_dims(state,axis = 0))
        action_sample = tf.random.categorical(logits = logits_actions,num_samples = 1)[0,0]
        action_logprobs_t = tf.nn.log_softmax(logits_actions)
        state, reward, done = tf_env_step(action_sample)

        # store results for later
        action_logprobs = action_logprobs.write(t, action_logprobs_t[0, action_sample])
        critic_values = critic_values.write(t, critic_value[0,0])
        rewards = rewards.write(t, reward)
        if tf.cast(done, tf.bool):
            break
    rewards = rewards.stack()
    action_logprobs = action_logprobs.stack()
    critic_values = critic_values.stack()
    return rewards, action_logprobs,critic_values

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def compute_loss(action_logprobs:tf.Tensor,returns:tf.Tensor,critic_values:tf.Tensor) -> tf.Tensor:
    actor_loss = -tf.math.reduce_sum(action_logprobs*tf.stop_gradient(returns))
    critic_loss = huber_loss(critic_values, returns)
    return actor_loss+critic_loss 

def train_step(
    start_state: tf.Tensor, 
    ac_model: tf.keras.Model, 
    ac_optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""
  state = start_state
  rewards = tf.TensorArray(dtype =tf.int32, size = 0, dynamic_size = True)
  gamma_tf = tf.constant(gamma)
  P_tf = tf.constant(1.0)
  lmd = 0.5
  lmd_tf = tf.constant(lmd)
  for t in tf.range(max_steps_per_episode):
    traces = [] 
    with tf.GradientTape() as tape:
      logits_actions, cval = ac_model(tf.expand_dims(state,axis = 0))
      action_sample = tf.random.categorical(logits = logits_actions,num_samples = 1)[0,0]
      action_logprob = tf.nn.log_softmax(logits_actions)[0, action_sample]
      next_state, reward, done = tf_env_step(action_sample)
      _, cval_next = ac_model(tf.expand_dims(next_state,axis = 0))
      td = tf.cast(reward,tf.float32)+gamma_tf*cval_next - cval
      a_loss = -action_logprob*tf.stop_gradient(td)*P_tf
      c_loss = huber_loss(tf.cast(reward,tf.float32)+gamma_tf*cval_next, cval)
      loss = a_loss+ c_loss
    grads_init = tape.gradient(loss, ac_model.trainable_variables)
    
    if len(traces) == 0:
      traces = [tf.zeros_like(grad_val, dtype = tf.float32)
                  for grad_val in grads_init]
    traces = [gamma*lmd_tf*tr + grad_val \
                for tr, grad_val in zip(traces, grads_init)]
    grads = [tf.reshape(td*tr,tr.shape) for tr in traces]

    ac_optimizer.apply_gradients(zip(grads, ac_model.trainable_variables))
    rewards = rewards.write(t, reward)
    if tf.cast(done, tf.bool):
      break
    state = next_state
    P_tf = P_tf*gamma_tf

  episode_reward = tf.math.reduce_sum(rewards.stack())
  return episode_reward

def create_nn(dnn_spec : DNNSpec_TF):
    return tf.keras.models.Sequential([tf.keras.layers.Dense(num_ins,activation=act_fn)
                for num_ins,act_fn in zip(dnn_spec.neurons, dnn_spec.activations)])




#################### ACTUAL LOOP ###################
class ActorCriticSharedNetwork(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      spec_nn:DNNSpec_TF):
    """Initialize."""
    super().__init__()

    self.shared = create_nn(spec_nn)
    self.actor = tf.keras.layers.Dense(num_actions)
    self.critic = tf.keras.layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.shared(inputs)
    return self.actor(x), self.critic(x)

spec_nn = DNNSpec_TF(
    neurons = [128],
    activations = ['relu'])


ac_agent = ActorCriticSharedNetwork(env.action_space.n,spec_nn)


ac_agent_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)


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
        initial_state, ac_agent, ac_agent_optimizer, gamma, max_steps_per_episode))

    running_reward = episode_reward*0.01 + running_reward*.99
    rew_hist.append(episode_reward)
    t.set_description(f'Episode {i}')
    t.set_postfix(running_reward=running_reward,
        episode_reward=episode_reward)

    # Show average episode reward every 10 episodes
    if i % 250 == 0:
      np.savetxt('save_eps_ac_lambda_2n5',np.array(rew_hist))

    if running_reward > reward_threshold:  
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
