# Reinforcement Learning with Human Feedback
- Purpose: Alignment of the model with human values. Making the model:
  - More helpful.
  - Not causing harm: Using toxic language in completions. Reply in combative and aggressive voices. Provide detailed information about dangerous topics.
  - Less misinformation: Hallucinate, answer a question it doesn't know confidently.

## Reinforcement Learning
![Tic-Tac-Toe](https://levendlee.files.wordpress.com/2023/12/llm_3-1-1.png)
- *Objective*: Maximize reward received for actions.
- *Policy*: LLM itself.
- *Environment*: The context window. The space in which text can be entered via a prompt.
- *State*: Text in the context window.
- *Action*: Generating text.
- *Reward*: How closely the model output aligns with the human preferences.
- *Rollout*: The sequence pf states &amp; actions in the process of fine-tuning.

## Reward Model
A model to assess the alignment of a completion, to be used during RLHF training.

## How to do RLHF
- Have human labelers rank completion on helpfulness, harm, etc.
- Convert rankings to pairwise training data.
  - For example, less helpful completion using 0 as no reward and more helpful information using 1 as rewarded.
- Train reward model, the model returns a score on the alignment of the completion.
  - We can use the logits before the probabilities output.
- Iteratively fine-tune the model on the dataset. Update LLM weights based on the rewards.

## Reward hacking
![Avoid reward hacking](https://levendlee.files.wordpress.com/2023/12/llm_3-2.png)
- *Problem*: The model is biased towards reward model, and outputs aligned but not relevant results. Similar to overfitting.
- *Solution*: Add a regularize, a reference model that stays frozen. Adds a penalty term on their difference, for example the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

## KL divergence
$latex D_{KL}(P||Q) = \sum{P(x)log(\frac{P(x)}{Q(x)})}$ 


## Proximal Policy Optimization (PPO)

- Phase 1. Calculate loss of value function
$latex L^{VF} = \frac{1}{2}||V_{\theta}(s) - (\sum_{t=0}^T\gamma^t{r_t} | s_0=s)||_2^2$
  - $latex V_{\theta}(s)$: Value function. Estimated future rewards.
  - $latex (\sum_{t=0}^T\gamma^t{r_t} | s_0=s)$: Known future total reward.

- Phase 2. Calculate loss of policy function
$latex L^{Policy} = min(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)})\cdot\hat{A_t}, clip(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon)\cdot\hat{A_t})$
$latex \pi_\theta$ Model's probability distribution over tokens.
  - $latex \pi_\theta(a_t|s_t)$: Probabilities of next token on the updated LLM.
  - $latex \pi_{\theta_{old}}(a_t|s_t)$: Probabilities of next token on the initial LLM.
  - $latex dot\hat{A_t}$: Advantage term.
  - $latex \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1 - \epsilon, 1 + \epsilon$: Trust region with original and updated output close.

- Phase 3. Calculate entropy loss
$latex L^{ENT} = entropy(\pi_\theta(\cdot | s_t))$

- Combined together:
$latex L^{PPO} =  L^{VF} + c_1L^{Policy} + c_2L^{ENT}$

## Consitutional AI
- Consitution is a set of prompts describing the principles the model has to follow.
- Read teaming: Human construct prompts that elicits harmful or unwanted responsed.

## Reinforcement Learning with AI feedback
![RLAIF](https://levendlee.files.wordpress.com/2023/12/llm_3-3.png)

# LLM Powered Applications
![LLM Lifecycle](https://levendlee.files.wordpress.com/2023/12/llm_3-4.png)

- RAG: Retrieval-augmented generation. Grounding the model on external information. Bard is doing something like this.
- Chain-of-thought planing: Asks the model to show their work. Helps the model deal with more complex math problems.
- Program aided language (PAL) models: Have the LLM generate completions where reasoning steps are accompanied by computer code.
- ReAct: Combining reasoning and action. Shows a LLM through structured examples how to reason through a problem and decide on actions to take.
