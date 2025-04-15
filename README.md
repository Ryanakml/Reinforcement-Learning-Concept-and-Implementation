# Reinforcement Learning Frozen Lake Visualization

This web application demonstrates reinforcement learning concepts through an interactive visualization of the Q-learning algorithm applied to the Frozen Lake environment.

## Features

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
2. Go to the "Training & Visualization" tab and click "Train Agent"
3. Observe the learning metrics and visualizations
4. Switch to the "Interactive Mode" tab to test the agent's performance
5. Control the agent manually or let it follow the learned policy

Experiment with different parameters to see how they affect the agent's learning process!