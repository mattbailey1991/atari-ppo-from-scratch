# atari-ppo-from-scratch

Implementation of the PPO (Proximal Policy Optimization) algorithm from scratch to learn to play Atari games. Follows the tutorial from https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ 🕹️

![Image](https://github.com/user-attachments/assets/10ec8a95-8f7b-4737-bb24-15274250264b)

## Description

This project provides a clean and understandable implementation of the PPO (Proximal Policy Optimization) algorithm using Python. The goal is to train an agent to play Atari games directly from pixel inputs. 

**What is PPO?**

Proximal Policy Optimization (PPO) is a policy gradient reinforcement learning algorithm. It's designed to be more stable and sample-efficient than traditional policy gradient methods like REINFORCE.  PPO aims to improve the policy iteratively by taking small steps in policy space, ensuring that the new policy doesn't deviate too far from the old policy. This is achieved through a clipped surrogate objective function.

**Key Concepts of PPO:**

*   **Policy Gradient:** PPO is a policy gradient method, meaning it directly optimizes the policy (the agent's behavior) instead of learning a value function first.
*   **Actor-Critic:** PPO typically uses an actor-critic architecture. The *actor* is the policy that determines the agent's actions, and the *critic* estimates the value of being in a particular state.
*   **Surrogate Objective Function:** PPO uses a surrogate objective function to approximate the expected return of the policy. This function is easier to optimize than the true expected return.
*   **Clipping:** The core innovation of PPO is the clipping mechanism. It limits the change in the policy by clipping the probability ratio between the new and old policies. This prevents the policy from changing too drastically in a single update, leading to more stable training.
*   **Advantage Function:** PPO often uses an advantage function to reduce variance in the policy gradient estimate. The advantage function estimates how much better an action is compared to the average action in a given state.  This implementation uses Generalized Advantage Estimation (GAE).

**PPO Formulas:**

1.  **Probability Ratio:**

    ```
    r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
    ```

    where:
    *   `π_θ(a_t | s_t)` is the probability of taking action `a_t` in state `s_t` under the current policy `θ`.
    *   `π_θ_old(a_t | s_t)` is the probability of taking action `a_t` in state `s_t` under the old policy `θ_old`.

2.  **Clipped Surrogate Objective Function:**

    ```
    L(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1 - ε, 1 + ε) * A_t)]
    ```

    where:
    *   `A_t` is the advantage function at time step `t`.
    *   `ε` is a hyperparameter that controls the clipping range (e.g., 0.2).
    *   `clip(x, min, max)` clips the value of `x` to be within the range `[min, max]`.

3.  **Generalized Advantage Estimation (GAE):**

    ```
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
    ```

    where:
    *   `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the temporal difference (TD) error.
    *   `r_t` is the reward at time step `t`.
    *   `γ` is the discount factor.
    *   `λ` is the GAE parameter (0 <= λ <= 1).  A value of `λ = 1` reduces to TD(λ), and `λ = 0` reduces to TD(0).
    *   `V(s)` is the value function, estimating the expected return from state `s`.

    A more efficient, recursive calculation of GAE is often used:

    ```
    A_t = δ_t + γλA_{t+1}
    ```

    with `A_T = δ_T` for the last timestep `T`.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/mattbailey1991/atari-ppo-from-scratch.git
    cd atari-ppo-from-scratch
    ```

2.  Install the required Python packages:

    ```bash
    pip install torch gymnasium numpy
    ```

## Key Features

*   **PPO Implementation:** A clear and concise implementation of the PPO algorithm.
*   **Atari Environment:** Designed to work with Atari environments from the Gymnasium library.
*   **Customizable:** Easily adaptable to different Atari games and hyperparameters.
*   **From Scratch:** Built from the ground up for educational purposes.
