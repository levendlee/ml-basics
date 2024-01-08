# Unit 6 Actor-Critic Methods with Robotics Environments
## Intro
- Value-based method
  - Q-learning method
- Policy-based method
  - Policy-gradient method
    - Use Monte-Carlo sampling to estimate return as cannot calculate for all trajectories.
    - Use lots of samples as trajectories can lead to different returns with high variance.
    - Which causes slower training as lots of samples are needed.
- Actor-Critic methods
  - *An Actor* that controls **how our agent behaves**. (Policy-based)
  - *A Critic* that measures **how good the take action** is. (Value-based)
  - Will study one of the hybrid methods, Advantage Actor Critic (A2C).
  - The high level idea is **using a value function (critic) to replace Monte-Carol sampling**.

## Advantage Actor-Critic (A2C)
### The Actor-Critic Process
- *Actor*, a policy function parameterized by $\theta$: $\pi_\theta(s)$.
- *Critic*, a value function parameterized by $w$: $\hat{q}_w(s, a)$.

Process:
- 0. At timestamp $t$, current state $S_t$ from environment and pass it to actor and critic.
- 1. Actor (policy function) takes input $S_t$ and outputs an action $A_t$.
- 2. Critic (value function) takes input $S_t, A_t$ and outputs an a Q-value $\hat{q}_w(S_t, A_t)$.
- 3. The action $A_t$ results in a new state $S_{t+1}$ and a new reward $R_{t+1}$.
- 4. Actor updates parameters using Q-value. Produces new action $A_{t+1}$.
  - $\Delta \theta = \alpha \nabla_\theta (log \pi_\theta(s_t, a_t)) \hat{q}_w(s_t, a_t)$.
  - Here we use $\hat{q}_w(s_t, s_t)$ to approximate accumulative rewards to trajectories. Still the idea of Mote-Carol error.
- 5. Critic updates parameters.
  - $\Delta w = \beta(R(s, a)+\gamma\hat{q}_w(s_{t+1},a_{t+1})-\hat{q}_w(s_t,a_t))\nabla_w\hat{q}_w(s_t, a_t)$.
  - $R(s, a)+\gamma\hat{q}_w(s_{t+1},a_{t+1})-\hat{q}_w(s_t,a_t)$: Temproral-difference error.
  - $\nabla_w\hat{q}_w(s_t, a_t)$: Gradient of value function.

### Adding Advantage in Actor-Critic (A2C)
Measures how taking that action at a state is better compared to the average value of the state. Extra rewards. Push in direction if more, opposite direction if less. $A(s,a) = Q(s,a) - V(s)$.

Use it to replace action value function. $\hat{q}_w(S_t, A_t)$.

Use TD error as a good estimator of the advantage function $A(s_t, a_t) = r + \gamma V(s_{t+1}) - V(s_t)$. $r$ is the immediate return. $V(s_{t+1})$ is the average value of the next state.
