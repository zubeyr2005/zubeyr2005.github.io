---
title: "Guiding Emergence in RL (Part 1): Acceleration via Blind Dynamics?"
date: 2025-03-29 15:00:00 +0000
categories: [Reinforcement Learning, Emergence, Dynamical Systems]
tags:
  [
    Reinforcement Learning,
    Emergence,
    Dynamical Systems,
    PPO,
    Reaction-Diffusion,
  ]
toc: false
math: true
---

_This post is the first in a series exploring methods to guide or accelerate the emergence of complex behaviours in reinforcement learning._

A key theme in modern AI research is **emergence**: sophisticated capabilities arising from training agents on relatively simple objectives. This has been observed in LLMs developing reasoning pathways ([link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)) and in reinforcement learning (RL), where agents discover complex policies guided solely by reward signals. The "Reward is Enough" hypothesis posits that maximising reward in complex environments inherently necessitates the development of diverse cognitive abilities ([link](https://www.sciencedirect.com/science/article/pii/S0004370221000862)).

Particularly for tasks with **verifiable objectives** (mathematics and code), online RL algorithms like PPO seem adept at forcing models beyond mere imitation towards learning the underlying _process_ for achieving the goal ([link](https://kalomaze.bearblog.dev/why-does-grpo-work/)). This focus on process appears crucial for robust, generalisable skill acquisition.

This process discovery mirrors aspects of natural evolution, which, operating under significant information constraints (e.g., limited genome size), produces solutions of remarkable efficiency and elegance – sometimes described as "mechanisms too simple for humans to design" ([link](https://www.lesswrong.com/posts/6hDvwJyrwLtxBLHWG/mechanisms-too-simple-for-humans-to-design)).

Nature often leverages local, pattern-forming rules, mathematically described by frameworks like reaction-diffusion (RD) systems, to generate intricate structures from simple initial conditions. This prompts the question:

#### Can we accelerate RL's discovery of potent, perhaps elegantly simple, emergent processes?

Let's see!

## Reaction-Diffusion Systems: A Quick Primer on Nature's Pattern Engine

Reaction-diffusion (RD) systems offer a powerful mathematical lens for understanding how complex spatial patterns can spontaneously emerge from simple, local interactions. They fundamentally model how the concentrations ($u$) of one or more substances evolve over time ($t$) and space due to two processes:

1.  **Reaction ($f(u)$):** Local transformations or interactions between substances.
2.  **Diffusion ($D \nabla^2 u$):** The tendency of substances to spread out from areas of high concentration to low concentration, represented mathematically by the Laplacian operator $\nabla^2$ scaled by a diffusion coefficient $D$.

The general form is often written as:

$$
\frac{\partial u}{\partial t} = D \nabla^2 u + f(u)
$$

A widely studied example is the **Gierer-Meinhardt activator-inhibitor model**:

$$
\frac{\partial a}{\partial t} = D_a \nabla^2 a + \rho_a \frac{a^2}{h} - \mu_a a
$$

$$
\frac{\partial h}{\partial t} = D_h \nabla^2 h + \rho_h a^2 - \mu_h h
$$

Here:

- $a(x,t), h(x,t)$ are the concentrations at position $x$ and time $t$.
- $D_a, D_h$ are diffusion coefficients (often $D_h > D_a$ for stable patterns).
- $\rho_a, \rho_h$ are production/synthesis rates. The non-linear term $\frac{a^2}{h}$ indicates self-catalysis of $a$ (autocatalysis) and activation of $h$ production by $a$.
- $\mu_a, \mu_h$ are decay or removal rates.

Under certain conditions, small random fluctuations can be amplified, leading to spontaneous, stable spatial patterns like spots or stripes (Turing patterns) from an initially uniform state.

The conceptual modification I do involves introducing an additional "guidance" term that directly amplifies the existing state. If you were to add this directly to the continuous PDE system (which is _not_ simulated, is too expensive), it might look conceptually like this:

$$
\frac{\partial a}{\partial t} = D_a \nabla^2 a + \rho_a \frac{a^2}{h} - \mu_a a + \gamma \cdot \text{sign}(a)
$$

$$
\frac{\partial h}{\partial t} = D_h \nabla^2 h + \rho_h a^2 - \mu_h h + \gamma \cdot \text{sign}(h)
$$

Here, $\gamma$ represents a fixed guidance strength, and the $\text{sign}()$ function pushes positive concentrations further positive and negative deviations (if allowed) further negative. This term is "blind" as $\gamma$ is fixed and the term only depends on the local state ($a$ or $h$), not on any measure of task success (reward). The idea is to investigate if such a simple amplification mechanism can influence pattern formation or system dynamics in a way that benefits learning.

---

## My Implementation: A Simplified, Guided Dynamical System

Simulating full RD PDEs within an RL loop is complex. So, I use a simpler **RD-inspired dynamical system** integrated into the agent's model, functioning like a set of coupled ODEs.

1.  **Structure:** The system maintains internal state matrices for "activator" ($a$) and "inhibitor" ($h$). These matrices have dimensions related to aspects of the observation space and the number of possible actions. These internal states persist across time steps within an episode.
2.  **Dynamics:** These internal state matrices ($a, h$) are updated at each step using rules that include:
    - _Simplified "Diffusion":_ A coupling term that creates interaction between different components of the internal state matrices.
    - _Local "Reactions":_ Terms inspired by RD reaction kinetics applied to the $a$ and $h$ matrices.
    - **Fixed Blind Guidance:** The core addition – a term $\gamma \cdot \text{sign}(a)$ (and similarly for $h$) is added to the update. $\gamma$ is a predetermined, fixed guidance strength. This term amplifies the current sign of the internal states, operating solely on that internal information, without reference to the external reward.
    - _State Influence:_ The current observation from the environment influences which part of the internal state is most relevant or emphasized in the current step.

## Integration with PPO

This internal dynamical system is integrated with a standard PPO agent:

![Bird's eye view](/assets/diagram.png)

- The environment provides the current state $s_t$ and reward $r_t$.
- $s_t$ is processed by the main PPO policy network to produce initial action logits.
- In parallel, the RD-inspired system updates its internal state matrices ($a, h$) based on its dynamics, including the blind guidance term. It then produces an output vector based on its current internal state and influenced by $s_t$.
- **Blending:** This output vector from the RD system is combined with the PPO network's logits (e.g., via a weighted average) to produce the final logits used for action selection.
- The agent selects action $a_t$ based on these blended logits.
- PPO uses the reward $r_t$ to update the parameters $\theta$ of the main policy network. The parameters defining the RD system's dynamics (diffusion rates, reaction rates, $\gamma$) remain fixed.

Here, the RD-inspired system now acts more like an RNN hidden state, evolving its patterns over time to guide action selection.

### Experiments: CartPole and its Stateless Variant

For this simple tweak, I've tested it on two environments:

1.  **CartPole-v1:** Standard benchmark with full state observability.
2.  **Stateless CartPole:** Harder version with partial observations (position and angle only), requiring inference or memory.

**(Standard CartPole)**

![Standard CartPole performance comparison](/assets/image.png)

On the standard task, guidance had an immediate impact, with higher $\gamma$ values leading to much faster initial learning comparerd to the baseline.

**(Stateless CartPole)**

![Stateless CartPole performance comparison](/assets/image2.png)

Here, the baseline and low-guidance agents learned at a similar rate. The instability from guidance likely stems from the guidance being blind and not intrisically linked to reward.

---

## Conclusion and Next Steps

This initial investigation explored whether incorporating fixed, RD-inspired internal dynamics could accelerate reinforcement learning. The results suggest this approach is a double-edged sword: while these dynamics, acting as a persistent internal state, did accelerate learning in some tests compared to the baseline, the "blind" amplification (unaware of reward) led to significant instability, especially in partially observable settings.

The core issue is of course the complete decoupling of the internal dynamic updates from the reward signal. While structuring an agent's internal computations holds promise, doing so without reference to the agent's objective proved problematic.

Therefore, the natural next step is to explore methods that **intrinsically link reward** with the agent's internal dynamics. Future work (in the coming weeks/months) will focus on developing reward-aware guidance mechanisms, potentially through learning the dynamic's parameters, using reward to directly modulate the dynamics' stability, or adapting guidance strength based on performance.
