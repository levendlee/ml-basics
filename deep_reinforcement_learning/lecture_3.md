# Unit 3. Deep Q-Learning with Atari Games
## From Q-Learning to Deep Q-Learning
- Tabular Method: Type of problem in which the state and action are small enough to approximate value functions to be represented as arrays and tables.

- Deep Q-Learning: Approximate Q-value with a neural network.

## Deep Q-Network
### Preparing the input and temporal limitation
Resolves temporal limitations through combining frames.

### Deep Q Algorithm
Compares Q-value prediction and Q-target and uses gradient descent to update the weights of our deep Q-Network to approximate our Q-values better.

- Q-Target: $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$
- Q-Loss: $R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$

#### Two phases
- Sampling: Perform actions and store the observed experience tuples in a reply memory.
- Training: Select a small batch of tuples **randomly**(??) and learn from this batch using gradient descent update step.

#### Instability
Combining a non-linear Q-value function (neural network) and bootstrapping (when we update targets with existing estimates and not an actual return).

Solutions:
- *Experience Replay* to make more efficient use of experiences.
- *Fixed Q-Target* to stabilize the training.
- *Double Deep Learning* to handle the problem of the overestimation of Q-values.

#### Experience Replay
Initialize a replay memory buffer *D* with capacity *N* (hyperparameter). We then store experiences in the memory and sample a batch of experiences to feed during training.

- Make more efficient use of experiences during training. **Reuse** during training. Learn from same experiences multiple times.
- **Avoid forgetting** previous experiences (aka catastrophic interference, or catastrophic forgetting).

#### Random Sampling
Reduce the correlation between experiences. Avoid values from oscillating or diverging catastrophically.

#### Fixed Q-Target to stabilize the training
Because the use of Bellman equation, we have a TD target and and Q-Value are both calculated based on **the same weights**, significant correlation between them causing significant oscillation in training.

- Use a separate network with fixed parameters for estimating the TD target.
- Copy the parameters from our Deep Q-Network every C steps to update the target network.

#### Double DQN

Double Deep Q-Learning neural networks, handles the problem of the overestimate of Q-values.

Calculating TD target: how to make sure the best action for the next state is the action with the highest Q-value? We don't have enough information about the best action to take at the beginning.

- Use our DQN network to select the best action to take for the next state.
- Use our Target network to calculate the target Q value of taking that action at the next state.
