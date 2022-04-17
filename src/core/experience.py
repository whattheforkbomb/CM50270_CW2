from collections import namedtuple, deque
from typing import Union
import gym

from a2c.agent import BaseAgent

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))

class ExperienceBuffer():
    """
    A basic experience buffer. Provides information about the trajectory from epsiodes. 
    
    Parameters:
    - env (gym.Env or list[gym.Env]) environment(s) to use
    - agent (BaseAgent) - agent to use
    - buffer_size (int) - number of samples in the buffer per environment
    - steps_delta (int) - number of steps between experience items
    """
    def __init__(self, env: Union[gym.Env, list[gym.Env]], agent: BaseAgent, buffer_size: int = 2, steps_delta: int = 1) -> None:
        assert isinstance(env, (gym.Env, list))
        assert isinstance(agent, BaseAgent)
        
        self.agent = agent
        self.buffer_size = buffer_size
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []

        if isinstance(env, list):
            self.pool = env
        else:
            self.pool = [env]

    def __iter__(self) -> tuple[deque]:
        states, agent_states, histories, cur_rewards, cur_steps, env_lens = self.__split_observations()
        
        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input, states_indices, actions = self.__set_states()
            
            if states_input:
                sa_pairs, new_states = self.agent(states_input, agent_states)

                for idx, action in enumerate(sa_pairs):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_states[idx]
            
            grouped_actions = self.__group_list(actions, env_lens)

            global_offset = 0
            for idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                next_state, reward, done, _ = env.step(action_n[0])
                next_state_n, reward_n, done_n = [next_state], [reward], [done]

                for offset, (a, ns, r, d) in enumerate(zip(action_n, next_state_n, reward_n, done_n)):
                    # Update values
                    idx = global_offset + offset
                    state = states[idx]
                    history = histories[idx]
                    cur_rewards[idx] += r
                    cur_steps[idx] += 1

                    if state is not None:
                        history.append(Experience(state=state, action=a, reward=r, done=d))
                    if len(history) == self.buffer_size and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    
                    states[idx] = ns

                    if done:
                        # Handle short episodes
                        if 0 < len(history) < self.buffer_size:
                            yield tuple(history)
                        # Generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        
                        cur_rewards, cur_steps, agent_states, history = self.__reset_vals(cur_rewards, cur_steps, agent_states, history, idx)

                global_offset += len(action_n)
            iter_idx += 1

    def __set_states(self, states: list, actions: list) -> tuple:
        """
        Creates a list of state inputs and indices given an existing set of states and actions.
        
        Parameters:
        - states (list) - a list of existing states
        - actions (list) - a list of empty actions

        Returns:
        State inputs (list), and state indices (list), and the updated actions (list).
        """
        states_input, states_indices = [], []
        for idx, state in enumerate(states):
            if state is None:
                # Sample an action
                actions[idx] = self.pool[0].action_space.sample()
            else:
                states_input.append(state)
                states_indices.append(idx)
        return states_input, states_indices, actions

    def __split_observations(self) -> tuple:
        """
        Obtains the observations from the environment pool and divides them into respective lists.
        
        Returns:
        States (list), agent states (list), histories (deque), current rewards (list), 
        current steps (list), and environment observation lengths (list).
        """
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens: list[int] = []
        for env in self.pool:
            obs = env.reset()
            obs_len = 1
            states.append(obs)
            env_lens.append(obs_len)
            
            for _ in range(obs_len):
                histories.append(deque(maxlen=self.buffer_size))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())
        return states, agent_states, histories, cur_rewards, cur_steps, env_lens

    def __reset_vals(self, cur_rewards: list, cur_steps: list, agent_states: list, history: deque, idx: int) -> None:
        """Resets core parameters for next iteration."""
        self.total_rewards.append(cur_rewards[idx])
        self.total_steps.append(cur_steps[idx])
        cur_rewards[idx] = 0.0
        cur_steps[idx] = 0
        agent_states[idx] = self.agent.initial_state()
        history.clear()
        return cur_rewards, cur_steps, agent_states, history

    @staticmethod
    def __group_list(items: list, lengths: list[int]) -> list[list]:
        """
        Unflattens a list of items by its size.

        Parameters:
        - items (list) - list of items
        - lengths (int) - list of integers

        Returns:
        - A list of lists of items grouped with its lengths
        """
        grouped = []
        offset = 0
        for len in lengths:
            grouped.append(items[offset:offset + len])
            offset += len
        return grouped

class FirstLastExpBuffer(ExperienceBuffer):
    """An experience replay buffer that focuses on the first and last set of experiences. Used in A2C."""
    def __init__(self, env: Union[gym.Env, list[gym.Env]], agent: BaseAgent, gamma: float, buffer_size: int = 1, steps_delta: int = 1) -> None:
        super().__init__(env, agent, buffer_size + 1, steps_delta)
        self.gamma = gamma
        self.steps = buffer_size

    def __iter__(self) -> namedtuple:
        for exp in super().__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                items = exp
            else:
                last_state = exp[-1].state
                items = exp[:-1]
            
            total_reward = 0.
            for item in reversed(items):
                total_reward *= self.gamma + item.reward
            
            yield ExperienceFirstLast(
                state=exp[0].state, 
                action=exp[0].action,
                reward=total_reward,
                last_state=last_state
            )
