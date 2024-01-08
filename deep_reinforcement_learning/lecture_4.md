# Unit 4. Policy Gradient with PyTorch

## Value-based, Policy-based, and Actor-critic methods

- Value based
  - Learn a value function leading to an optimal policy. 
  - Minimize the predicted and target value. 
  - Generate policy directly from value function.

- Policy based
  - Learn to approximate $latex \pi^\*$ without learning a value function. 
  - Parameterize the policy.
  - e.g. Stochastic Policy: $latex \pi_{\theta}(s) = P[A|S;\theta]$
  - Define an objective function $latex J(\theta)$, expected cumulative reward, and want to find the value $latex \theta$ that maximizes the objective function.

- Actor-critic
  - A combination of both.

## Difference between policy-based and policy-gradient methods

Policy-gradient methods is a subclass of policy-based methods. 

- In policy-based methods, the optimization is most of the time *on-policy* since for each update, we only use data (trajectories) collected by our most recent version of policy.

Difference lies in how to optimize the parameters:
- In *policy-based* methods, search directly for optimal policy. Optimize the parameter **indirectly** by maximizing the local approximation of the object function with *hill climbing, simulated annealing or evolution strategies*.
- In *policy-gradient* methods, also search directly for optimal policy. Optimize the parameter **directly** by performing **gradient ascent** on the performance of the objective function.

### Advantages

- Simplicity.
- Learn a stochastic policy.
  - Don't need to implement exploration/exploitation trade-off by hand.
  - Get ride of perceptual aliasing. (Two states seem the same but need different actions).
- More efficient in high-dimensional action spaces and continuous action spaces.
  - Deep Q-learning assigns a score for each possible action. But polcy gradients output a probability distribution over actions.
- Better convergence properties.
  - Smooth change without using `argmax` which is applied in value-based methods.

### Disadvantages

- Converge to a local maximum instead of a global optimum.
- Slower: Step by step. It can take longer to train.
- High variance. (To be discussed with actor-critic unit)

## Deeper dive into policy-gradient

### Policy Gradient Algorithm
  - Training Loop
    - Collect an episode with the policy.
    - Calculate the return (sum of rewards).
  - Update weights of the policy.
    - If positive return -> increase the possibility of each (state, action) pairs taken during the episode.
    - If negative return -> decrease.

### Objective Function
Performance of the agent given a trajectory (state action sequence without considering reward), outputs the *expected cumulative reward*.

$latex J(\theta) = E_{\tau \sim \pi}[R(\tau)] = \sum_{\tau}P(\tau;\theta)R(\tau) = \sum_{\tau}[\prod_{t=0}P(s_{t+1}|s_t,a_t)\pi_{\theta}(a_t|s_t)]R(\tau)$

in which
$latex R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots$: Discounted cumulative reward of the trajectory.

### Optimize Objective Function
$latex \max_{\theta} J(\theta) = E_{\tau \sim \pi}[R(\tau)]$

$latex \theta \leftarrow \theta + \alpha * \nabla_{\theta}J(\theta)$

The problem:
  - **Can't calculate the true gradient of the objective function** $latex P(\tau;\theta)$ since it requires calculating the probability of each possible trajectory. Require a sample-based estimate.
  - **Can't differentiate the state distribution** (Markov Decision Process dynamics) $latex P(s_{t+1}|s_t,a_t)$ as we might not know about it.

The solution:
  - **Policy Gradient Theorem**: $latex \nabla_\theta J(\theta) = E_{\pi_\theta}[\nabla_\theta log \pi_\theta(a_t|s_t)R(\tau)]$
  - [Math proof](https://huggingface.co/learn/deep-rl-course/unit4/pg-theorem). The main tricks are:
    - Derivative log trick (likelihood ratio trick or reinforce trick):
       - $latex \nabla_x log f(x) = \frac{\nabla_x f(x)}{f(x)}$. 
       - Translate $latex \frac{\nabla_\theta P(\tau;\theta)}{P(\tau;\theta)}$ to $latex \nabla_\theta logP(\tau;\theta)$.
    - Use sampling to approximate distribution/expectation.
       - Translate $latex \sum_{\tau} P(\tau;\theta)\nabla_\theta P(\tau;\theta) R(\tau)$ to $latex \frac{1}{m} \sum_{i=1}^m \nabla_\theta P(\tau^{(i)};\theta) R(\tau^{(i)})$

### The Reinforcement algorithm (Monte Carlo Reinforce)

- Monte carol reinforce: Uses an estimate return from **an entire episode to update the policy parameter $latex \theta$.

- One trajectory: $latex \nabla_\theta J(\theta) \approx \hat{g} = \sum_{t=0} \nabla_\theta log(\pi_\theta)(a_t|s_t)R(\tau)$
- Multiple trajectories: $latex \nabla_\theta J(\theta) \approx \hat{g} = \frac{1}{m} \sum_{i=1}^m \sum_{t=0} \nabla_\theta log(\pi_\theta)(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)})$

- Intuitions
  - $latex \nabla_\theta log \pi_\theta(a_t|s_t)$ is the direction of **steepest increase of the (log) probability** of selecting action $latex a_t$ from state $latex s_t$. $latex R(\tau)$ is the scoring function.
  - If return is high, it will **push up the probabilities** of the (state, action) combinations.
  - Otherwise, **push down**.
  