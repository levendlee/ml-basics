# Unit 1. Introduction to Deep Reinforcement Learning

## Concept

Reinforcement learning is a framework for solving control tasks by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

## Framework

Agent/Environment + State/Action/Reward.
- RL loop outputs a sequence of state, action, reward, and next state.
- RL goal is to maximize the cumulative reward, called **expected return**.

### Reward hypothesis
All goals can be described as the maximization of the expected return (expected cumulative reward).

### Markov Property

RL process is called **Markov Decision Process(MDP)**. It implies that our agent needs only the current state to decide what action to take and not the history of all the states and actions they took before.

### Observations/States Space

- State: Complete description of the state of the word (no hidden information).
- Observation: Partial description of the state of the word.

### Action Space

- Discrete: Finite possible actiions.
- Continuous: Infinite possible actions.

### Rewards and discounting

$R(\tau) = \sum_{k=0}^{\infty}{\gamma^kr_{t+k+1}}$

R: cumulative reward.
Gamma: Discount rate. 0.95-0.99. Higher for value long-term. Lower for value short-term.

### Type of tasks

- Episodic: Starting point and ending point (a terminal state).
- Continuing: Task that continue forever (no terminal state).

### Exploration/Exploitation Trade-off

- Exploration: Trying random actions in order to find more information about the environment.
- Exploitation: Using known information to maximize the reward.

## Main Approaches to resolve RL problems

Policy, the function that tells what action to take given the state we are in. It defines the agent's behavior at a given time.

## Policy-Based Methods
*Learn a function that inputs state and outputs action.*

Learn policy function directly. Teaching the agent to learn which **action** to take, given the current state.

This function will define a mapping from each state to the best corresponding action. Alternatively, it could define a probability distribution over the set of possible actions at that state.

- Deterministic: A policy that a given state will always return the same action. $a = \pi(s)$.
- Stochastic: A probability distribution over actions. $\pi(a|s) = P[A|s]$

## Value-Based methods
*Learn a function that input state and output expected discount accumulative reward.*
Learn a value function that maps a **state** to the expected value of being at that state. Teaching the agent to learn which state is more valuable and then take the action that leads to the more valuable state.

$v_{\pi}(s) = E_{\pi}[\sum_{k=0}^{\infty}{\gamma^kr_{t+k+1}}| S_t = s]$

### Traditional vs Deep Reinforcement Learning

- Traditional Q learning: Use Q table to find which action to take for each state.
- Deep Q learning: Use a NN to approximate Q table.
