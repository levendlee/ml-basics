# Unit 2. Introduction to Q-Learning

## Policy of value-based methods
Always need a policy to direct the agent taking acitions.
- In policy-based training, the optimal policy ($\pi^*$) is found by training the policy directly.
- In value-based training, finding an optimal value function (
$Q^*$ or 
$V^*$
) leads to having an optimal policy.
$\pi^*(s) = arg {max}_a Q^*(s, a)$

In value based method, use **Epsilon-Greedy Policy** that handles the exploration/exploitation trade-off.

## Two types of value functions

### The state-value function
For each state, outputs the expected return if the agent starts at that state and then follows the policy forever afterwards (for all future timestamps).
$V_{\pi}(s) = E_{\pi}[G_t|S_t = s]$

### The action-value function
For each state-action pair, the action-value function outputs the expected return if the agent starts in that state, takes that action, and then follows the policy forever.
$Q_{\pi}(s, a) = E_{\pi}[G_t|S_t=s, A_t=a]$

The problem is, *to EACH value of a state or a state-action pair, we need to sum all the rewards an agent can get if it starts at that state*.

## The Bellman Equation: Simplify value estimation
Recursive equation. Similar idea to dynamic programming.

$V_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma * V_{\pi}(S_{t+1})|S_t = s]$

## Monte Carlo vs Temporal Difference Learning

### Monte Carlo: learning at the end of episode
- Requires a complete episode of interaction before updating our value function.
- With Monte Carlo, we update the value function from a complete episode, and so we use *the actual accurate discounted return of this episode*.

$V(S_t) \leftarrow  V(S_t) + \alpha[G_t - V(S_t)]$

### Temporal Difference Learning: learning at each step
- Update at one interaction but estimate $G_t$ by adding $R_{t+1}$ and the discounted value of the next state.
- With TD Learning, we update the value function from a step, and we replace $G_t$, which unknown, with *an estimated return called TD target.*

$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

## Introducing Q Learning
### What is Q-Learning
- Off-policy.
- Value-based method. State-value/action value.
- Temporal difference (TD) approach.

Q(Quality)-Learning, algorithm to train Q-function, an action-value function that determines the value of being a particular state and taking a particular action.

$\pi^*(s) = arg \max_a Q^*(s,a)$

## Q-Learning algorithm
### Initialize Q-table
Initialize for each state-action pair. Most of the time, initialize with values of 0.

### Choose an action using the epsilon-greedy strategy
Handles exploration/exploitation trade-off.

With an initial values of $\epsilon = 1.0$:
- With probability $1 - \epsilon$: We do exploitation (agent selects the action with the highest state-action pair value).
- With probability $\epsilon$: We do exploration (trying random action).
**With training goes on, progressively reduce the epsilon value since Q-table gets better and better in its estimations.**

### Perform action $A_t$, get reward $R_{t+1}$ and next state $S_{t+1}$.

### Update $Q(S_t, A_t)$.
$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$

- $R_{t+1}$: Immediate reward.
- $\gamma \max_a Q(S_{t+1}, a)$: Discounted estimate optional Q-value of next stage by finding the action that maximizes the current Q-function at the next state.
- $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$: TD Target.
- $R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$: TD Error.


*This is not the epsilon-greedy policy, as the greedy policy, AKA. always take the action with the highest state-action value.* Start in a new session and then select action using epsilon-greedy policy again. **off-policy**.

### SARSA

[On-policy variant of Q-learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/).
$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$


### Off-policy vs On-policy

- Off-policy: Using a different policy for acting (inference) and updating (training).
- On-policy: Using the same policy for acting and updating.
