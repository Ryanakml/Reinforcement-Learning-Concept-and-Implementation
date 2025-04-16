# Reinforcement Learning Concepts
This repository contains a collection of reinforcement learning concepts implemented in Python. Each concept is presented in a separate directory, with a README explaining its purpose, implementation details, and usage. To learn the concepts, simply navigate to RL in Math.ipynb 

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*S1bubWcPjP_zR-0g4AiV5Q.png)


# Reinforcement Learning Frozen Lake Visualization

This web application demonstrates reinforcement learning concepts through an interactive visualization of the Q-learning algorithm applied to the Frozen Lake environment.

## Features

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*kms9wM4_Y9E7sKRj4kGS3g.png)

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*xJax9699g-XuW9YpIs5VNw.png)

- **Interactive Training**: Train a reinforcement learning agent with customizable parameters
- **Visual Q-value Representation**: Heatmap visualization of learned Q-values
- **Policy Visualization**: See the optimal actions learned by the agent
- **Interactive Mode**: Control the agent manually or let it follow the learned policy
- **Learning Metrics**: Track success rate and rewards during training
- **Educational Content**: Learn about reinforcement learning concepts

## Environment

In the Frozen Lake environment, an agent must navigate from the starting point to a goal without falling into holes. The lake surface can be slippery (optional), making the agent's movements uncertain.

- 🤖 Agent: The learning algorithm
- 🏁 Goal: The target destination
- 🕳️ Holes: Areas to avoid
- ⬜ Frozen surface: Safe to walk on

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Customizable Parameters

- **Learning rate (α)**: Controls how quickly the agent updates its knowledge
- **Discount factor (γ)**: Determines the importance of future rewards
- **Exploration rate (ε)**: Controls the exploration-exploitation trade-off
- **Slippery Environment**: Toggle to make the environment more challenging
- **Training Episodes**: Number of episodes to train the agent

## How to Use

1. Adjust the learning parameters in the sidebar

Slippery Environment = False : More accurate
Slippery Environment = True : More challenging

α = 0.1
γ = 0.95
ε = 1 
episodes = 5000

2. Go to the "Training & Visualization" tab and click "Train Agent"

3. Observe the learning metrics and visualizations

α = new more accurate
γ = new more accurate
ε = new more accurate
episodes = new more accurate

4. Switch to the "Interactive Mode" tab to test the agent's performance

5. Control the agent manually or let it follow the learned policy (Based on Training)

## Learning Process
Experiment with different parameters to see how they affect the agent's learning process!