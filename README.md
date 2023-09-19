# <p align="center">POLICY ITERATION ALGORITHM</p>

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

</br></br></br>

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
<p align="center">
<img width="550" src="https://github.com/ShafeeqAhamedS/RL_2_Policy_Eval/assets/93427237/e7af87e7-fe73-47fa-8bea-2040b7645e44"> </p>


## POLICY ITERATION ALGORITHM

The algorithm implemented in the **policy_iteration** is a method used to find the optimal policy in a Markov decision process (MDP). 
Here's a step-by-step explanation of the algorithm:

1. Initialize the policy **pi**. In this implementation, a random action is chosen for each state **s** in the MDP **P**. The initial policy is represented by the lambda function **pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s]**, where **random_actions** is a list of randomly chosen actions for each state.

2. Enter a loop that continues until the policy **pi** is no longer changing. This is determined by comparing the previous policy (**old_pi**) with the current policy computed in the loop.

3. Store the previous policy as **old_pi** for comparison later.

4. Perform policy evaluation using the function **policy_evaluation**. This step calculates the state-values (**V**) for each state **s** given the current policy **pi**. The state-values represent the expected cumulative rewards starting from state **s** following policy **pi** and discounting future rewards by a factor of **gamma**. The function **policy_evaluation** is called with the arguments **pi**, **P**, **gamma**, and **theta**.

5. Perform policy improvement using the function **policy_improvement**. This step updates the policy **pi** based on the current state-values **V**. The function **policy_improvement** is called with the arguments **V**, **P**, and **gamma**.

6. Check if the policy has converged by comparing the previous policy **old_pi** with the current policy **{s:pi(s) for s in range(len(P))}**. If they are the same for all states **s**, the loop is exited.

7. Return the final state-values **V** and the optimal policy **pi**.

To summarize, policy iteration iteratively improves the policy by alternating between policy evaluation and policy improvement steps until convergence is reached. The algorithm guarantees to find the optimal policy for the given MDP **P** with a discount factor **gamma**.

### POLICY IMPROVEMENT FUNCTION
```py
def policy_improvement(V, P, gamma=0.9):
    # Initialize a matrix to store action values for each state-action pair
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Loop through all states
    for s in range(len(P)):
        # Loop through all possible actions for the current state
        for a in range(len(P[s])):
            # Calculate the expected future rewards (action values) for each action
            for prob, next_state, reward, done in P[s][a]:
                # Update the action value for the current state-action pair
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    # Determine the new policy by selecting the action with the highest action value
    new_policy = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    # Return the new policy based on action values
    return new_policy
```

### POLICY ITERATION FUNCTION
```py
def policy_iteration(P, gamma=0.9, theta=1e-10):
    # Initialize a random policy by selecting random actions for each state
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
    
    while True:
        # Store the current policy for comparison
        old_pi = {s: pi(s) for s in range(len(P))}
        # Policy Evaluation: Compute the value function V under the current policy
        V = policy_evaluation(pi, P, gamma, theta)
        # Policy Improvement: Improve the policy based on the current value function
        pi = policy_improvement(V, P, gamma)
        # Check if the policy has converged (no change from the previous iteration)
        if old_pi == {s: pi(s) for s in range(len(P))}:
            break
    # Return the final value function and the optimal policy
    return V, pi
```

## OUTPUT:
### Adversarial Policy
<img width="1000" src="https://github.com/ShafeeqAhamedS/Rl_Exp_3_policy-iteration-algorithm/assets/93427237/e68df586-ba40-4b7c-838f-12f48bb5c590">

### Optimal Policy with Gamma = 0.90
<img width="1000" src="https://github.com/ShafeeqAhamedS/Rl_Exp_3_policy-iteration-algorithm/assets/93427237/845dc301-0361-4a3b-a204-8e78ac1e7dd3">

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.
