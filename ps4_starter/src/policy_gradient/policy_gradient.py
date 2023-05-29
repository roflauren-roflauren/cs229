"""
CS 229 Machine Learning
Question: Reinforcement Learning - Policy Gradient
"""
import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np


class Policy:
    def __init__(self, theta, alpha=0.001, weighting='full_trajectory'):
        """
        Constructs the policy gradient model.

        Args:
            theta (np.ndarray): Initial model parameters, having shape (state_dim,).
            alpha (float): The learning rate (default: 0.001).
            weighting (str): Specifies how episode rewards are used to weight gradients (default: 'full_trajectory').
        """
        self.theta = theta
        self.alpha = alpha

        if weighting == 'full_trajectory':
            self.weighting_fn = self.compute_weights_full_trajectory
        elif weighting == 'reward_to_go':
            self.weighting_fn = self.compute_weights_reward_to_go
        else:
            raise Exception('Invalid weighting function: {}'.format(weighting))

    def sigmoid(self, x):
        """
        Calculates the sigmoid activation function.

        Args:
            x (np.ndarray): Arbitrary input array to the sigmoid function.

        Returns:
            np.ndarray, where sigmoid is applied elementwise to each value in x.
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x))
        # *** END CODE HERE ***

    def policy(self, state):
        """
        Returns the probabilities of taking each action for a given state.

        Args:
            state (np.ndarray): State to be evaluated with shape (state_dim,)

        Returns:
            np.ndarray of shape (num_actions,) == (2,) containing [P(left | state), P(right | state)].
        """
        # *** START CODE HERE ***
        pi_zero = self.sigmoid(np.dot(self.theta, state))
        return np.array([pi_zero, 1 - pi_zero])
        # *** END CODE HERE ***
        
    def sample_action(self, state):
        """
        Samples a random action according to the model policy.
        Please use np.random.choice or np.random.uniform in your implementation.

        Args:
            state (np.ndarray): State to be evaluated with shape (state_dim,)

        Returns:
            Randomly sampled action, either 0 (representing left) or 1 (representing right).
        """
        # *** START CODE HERE ***
        action_probs = self.policy(state)
        action = np.random.choice([0,1], p=action_probs)
        return action         
        # *** END CODE HERE ***

    def grad_log_prob(self, state):
        """
        Calculates the gradient of the log probabilities.

        Args:
            state (np.ndarray): State to be evaluated with shape (state_dim,)

        Returns:
            np.ndarray of shape (num_actions, state_dim) == (2, state_dim) containing grad ln(P(left | state)) in the first row and grad ln(P(right | state)) in the second row
        """
        # *** START CODE HERE ***
        grad_pi_zero = (1 - self.sigmoid(np.dot(self.theta, state))) * state
        grad_pi_one  = -1 * (1 - self.sigmoid(-1 * np.dot(self.theta, state))) * state 
        return np.array([grad_pi_zero, grad_pi_one])
        # *** END CODE HERE ***

    def compute_weights_full_trajectory(self, episode_rewards):
        """
        Calculates cumulative rewards across the entire episode for vanilla policy gradient.

        >>> compute_weights_full_trajectory(np.array([r0, r1, r2]))
        np.array([r0 + r1 + r2, r0 + r1 + r2, r0 + r1 + r2])

        Args:
            episode_rewards (np.ndarray): The rewards observed over the episode with shape (episode_length,).

        Returns:
            np.ndarray of shape (episode_length,) containing cumulative rewards according to specification given above.
        """
        # *** START CODE HERE ***
        return np.full(episode_rewards.shape, np.sum(episode_rewards))
        # *** END CODE HERE ***

    def compute_weights_reward_to_go(self, episode_rewards):
        """
        Calculates cumulative rewards across the entire episode for the "reward-to-go" characterization.

        >>> compute_weights_reward_to_go(np.array([r0, r1, r2]))
        np.array([r0 + r1 + r2, r1 + r2, r2])

        Args:
            episode_rewards (np.ndarray): The rewards observed over the episode with shape (episode_length,).

        Returns:
            np.ndarray of shape (episode_length,) containing cumulative rewards according to specification given above.
        """
        # *** START CODE HERE ***
        weights = []
        for i in range(len(episode_rewards)): 
            weights.append(sum(episode_rewards[i:]))
        return np.array(weights)
        # *** END CODE HERE ***

    def update(self, episode_rewards, states, actions):
        """
        Updates the model parameters theta given the rewards, states, and actions with learning rate alpha.

        Args:
            episode_rewards (np.ndarray): The rewards observed over the episode with shape (episode_length,).
            states (np.ndarray): The states observed over the episode with shape (episode_length, state_dim).
            actions (np.ndarray): The actions observed over the episode with shape (episode_length,).

        Returns:
            None (This function does not return anything, it modifies theta in-place.)
        """
        cumulative_rewards = self.weighting_fn(episode_rewards)
        # *** START CODE HERE ***
        n_timesteps = len(actions)
        theta_grad = np.zeros(shape=self.theta.shape)
        for t in range(n_timesteps): 
            theta_grad += self.grad_log_prob(states[t])[actions[t]] * cumulative_rewards[t]
        self.theta = self.theta + self.alpha * theta_grad
        # *** END CODE HERE ***
        return # This function does not return anything


def run_episode(env, policy, render=False):
    """
    Runs the policy for a single episode.

    Args:
        env (gym.Env): The OpenAI Gym environment.
        policy (Policy): The linear policy being trained.
        render (bool): Whether to render the model and play the simulation for this episode.

    Returns:
        The total rewards observed over the episode, as well as the rewards, states, and actions taken.
    """
    state = env.reset()
    states, actions, rewards = [], [], []
    total_reward, done = 0, False

    while not done:
        if render:
            env.render()

        states.append(state)

        action = policy.sample_action(state)
        state, reward, done, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        actions.append(action)
    return total_reward, np.array(rewards), np.array(states), np.array(actions)


def train(
        env,
        alpha=0.001,
        weighting='full_trajectory',
        max_episodes=4000,
        render=False,
        render_every_n=100,
    ):
    """
    Trains a linear policy via policy gradient according to the training parameters specified.

    Args:
        env (gym.Env): The OpenAI Gym environment.
        alpha (float): The learning rate (default: 0.001).
        weighting (str): Specifies how episode rewards are used to weight gradients (default: 'full_trajectory').
        max_episodes: Maximum number of episodes to train for (default: 4000).
        render (bool): Whether to render the model and play the simulation during training (default: False).
        render_every_n (int): If render is True, render every n episodes (default: 100).

    Returns:
        The total rewards observed over each episode, as well as the learned policy.
    """
    state_dim = env.observation_space.shape[0]
    theta = np.random.randn(state_dim)
    policy = Policy(theta, alpha, weighting)

    episode_rewards = []
    for episode in range(max_episodes):
        render_this_episode = render and (episode % render_every_n == 0)
        total_reward, rewards, states, actions = run_episode(env, policy, render_this_episode)
        episode_rewards.append(total_reward)
        policy.update(rewards, states, actions)
        print("EP: " + str(episode) + " Score: " + str(total_reward) + " ", end="\r", flush=False)

    return episode_rewards, policy


def plot_util(rewards, weighting):
    """
    Plot utility.
    """
    plt.plot(range(len(rewards)), rewards, rasterized=True)
    plt.xlabel('# of Episode')
    plt.ylabel('Episode Total Reward')
    plt.ylim(-20, 220)
    plt.savefig('{}.png'.format(weighting))


def main(
        env_name='CartPole-v0',
        seed=0,
        alpha=0.001,
        weighting='full_trajectory',
        max_episodes=4000,
        render=False,
        render_every_n=100,
    ):
    """
    Main runner. Takes in command line arguments, trains a linear policy, and plots the results.

    Args:
        env_name (str): The name of the OpenAI Gym environment (default: 'CartPole-v0').
        seed (int): The seed to be applied for reproducibility (default: 0).
        alpha (float): The learning rate (default: 0.001).
        weighting (str): Specifies how episode rewards are used to weight gradients (default: 'full_trajectory').
        max_episodes (int): Maximum number of episodes to train for (default: 4000).
        render (bool): Whether to render the model and play the simulation during training (default: False).
        render_every_n (int): If render is True, render every n episodes (default: 100).
    """
    # Create environment and set seed for reproducibility.
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)

    # Train policy and retrieve performance across episodes.
    episode_rewards, policy = train(
        env=env,
        alpha=alpha,
        weighting=weighting,
        max_episodes=max_episodes,
        render=render,
        render_every_n=render_every_n,
    )

    # Plot results.
    plot_util(episode_rewards, weighting)


def parse_args():
    parser = argparse.ArgumentParser(description='Policy Gradient with Linear Policy')

    parser.add_argument('--alpha', type=float, default=0.001, help='The learning rate (default: 0.001).')
    parser.add_argument('--env_name', type=str, default='CartPole-v0', help='The name of the OpenAI Gym environment (default: CartPole-v0).')
    parser.add_argument('--max_episodes', type=int, default=4000, help='Maximum number of episodes to train for (default: 4000).')
    parser.add_argument('--render', type=bool, default=False, help='Whether to render the model and play the simulation during training (default: False).')
    parser.add_argument('--render_every_n', type=int, default=100, help='If render is True, render every n episodes (default: 100).')
    parser.add_argument('--seed', type=int, default=80, help='The seed to be applied for reproducibility (default: 80).')
    parser.add_argument('--weighting', type=str, default='full_trajectory', choices=['full_trajectory', 'reward_to_go'], help='Specifies how episode rewards are used to weight gradients (default: full_trajectory).')

    return vars(parser.parse_args())


if __name__ == '__main__':
    main(**parse_args())
