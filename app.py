import streamlit as st
import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# App configuration
st.set_page_config(page_title="RL Frozen Lake", page_icon="🧊", layout="wide")

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.Q = None
    st.session_state.rewards_history = []
    st.session_state.success_rate_history = []
    st.session_state.episode_count = 0
    st.session_state.state = None
    st.session_state.done = False
    st.session_state.path = []
    st.session_state.is_slippery = False
    st.session_state.training_complete = False

# Function to display grid with agent
def render_grid(state):
    # This function needs to be defined after environment initialization
    # but will be called later when the environment is ready
    if 'initialized' not in st.session_state or not st.session_state.initialized:
        return
    
    # Use environment from session state for consistency
    if not hasattr(st.session_state, 'env'):
        return
        
    grid_size = int(np.sqrt(st.session_state.env.observation_space.n))
    tiles = ""
    path_positions = set(st.session_state.path) if hasattr(st.session_state, 'path') else set()
    
    for i in range(grid_size):
        row = ""
        for j in range(grid_size):
            idx = i * grid_size + j
            
            # Determine tile type
            if idx == state:
                tile = '<div class="tile agent">🤖</div>'
            elif idx == grid_size * grid_size - 1:
                tile = '<div class="tile goal">🏁</div>'
            elif idx in [5, 7, 11, 12]:  # Hole positions in default FrozenLake 4x4
                tile = '<div class="tile hole">🕳️</div>'
            elif idx in path_positions and idx != state:
                # Show footprints for the path taken
                tile = '<div class="tile path">👣</div>'
            else:
                # Show Q-values as background intensity if available
                if hasattr(st.session_state, 'Q') and st.session_state.Q is not None:
                    q_val = np.max(st.session_state.Q[idx])
                    # Normalize q_val for color intensity (0 to 100)
                    intensity = min(100, max(0, int(q_val * 100)))
                    tile = f'<div class="tile empty" style="background-color: rgba(135, 206, 250, {intensity/100});"></div>'
                else:
                    tile = '<div class="tile empty"></div>'
            
            row += tile
        tiles += f'<div class="row">{row}</div>'

    st.markdown(f"""
    <style>
    .row {{
        display: flex;
        justify-content: center;
        margin: 2px;
    }}
    .tile {{
        width: 60px;
        height: 60px;
        border: 2px solid #ddd;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 28px;
        background-color: #eee;
        border-radius: 10px;
        margin: 2px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    .tile:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    .tile.agent {{ background-color: #ffd700; animation: pulse 1.5s infinite; }}  /* Yellow with pulsing animation */
    .tile.goal  {{ background-color: #90ee90; }}  /* Light green */
    .tile.hole  {{ background-color: #ff7f7f; }}  /* Light red */
    .tile.path  {{ background-color: #e6e6fa; }}  /* Light lavender for path */
    .tile.empty {{ background-color: #ffffff; }}
    
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }}
        70% {{ box-shadow: 0 0 0 10px rgba(255, 215, 0, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }}
    }}
    </style>
    <div style="display:flex; flex-direction:column; align-items:center;">
    {tiles}
    </div>
    """, unsafe_allow_html=True)

    # Display action probabilities for current state if Q-table exists
    if hasattr(st.session_state, 'Q') and st.session_state.Q is not None and not st.session_state.done:
        st.markdown("### Action Probabilities for Current State", help="Shows the likelihood of each action in the current state")
        q_values = st.session_state.Q[state]
        action_names = ["Left", "Down", "Right", "Up"]
        action_symbols = ['←', '↓', '→', '↑']
        
        # Convert Q-values to probabilities using softmax
        q_exp = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probabilities = q_exp / q_exp.sum()
        
        # Create a compact visualization
        best_action = np.argmax(q_values)
        
        # Create a more compact figure with adjusted size to prevent overlap
        fig, ax = plt.subplots(figsize=(5, 1.5))
        
        # Create a custom colormap from light blue to dark blue
        cmap = LinearSegmentedColormap.from_list(
            "prob_cmap", [(0, "#e6f2ff"), (0.5, "#99ccff"), (1, "#0066cc")]
        )
        
        # Create a 1x4 heatmap for the four directions
        sns.heatmap(
            np.array([probabilities]), 
            annot=np.array([action_symbols]),
            fmt='',
            cmap=cmap,
            cbar=False,
            linewidths=1.5,
            ax=ax,
            annot_kws={"fontsize":12, "weight":"bold"}
        )
        
        # Add probability values with adjusted position to prevent overlap
        for i, prob in enumerate(probabilities):
            ax.text(i + 0.5, 0.75, f"{prob:.2f}", 
                    ha="center", va="center", fontsize=8,
                    color="black" if prob < 0.7 else "white")
        
        # Highlight the best action with a border
        ax.add_patch(plt.Rectangle((best_action, 0), 1, 1, fill=False, edgecolor='red', lw=2))
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("Action Probabilities", fontsize=10)
        
        # Make the plot more compact
        plt.tight_layout()
        st.pyplot(fig)

# Sidebar configuration
st.sidebar.title("🧠 RL Parameters")
st.sidebar.markdown("### Environment Settings")
is_slippery = st.sidebar.checkbox("Slippery Environment", value=st.session_state.is_slippery)
if is_slippery != st.session_state.is_slippery:
    st.session_state.is_slippery = is_slippery
    st.session_state.initialized = False

# Add method selection option
st.sidebar.markdown("### Training Method")
training_method = st.sidebar.radio(
    "Select training method",
    ["Q-Learning", "Value Iteration"],
    help="Choose the reinforcement learning algorithm to train the agent"
)

st.sidebar.markdown("### Learning Parameters")

# Parameters for both methods
gamma = st.sidebar.slider("Discount factor (γ)", 0.01, 1.0, 0.95, help="Determines the importance of future rewards")

# Method-specific parameters
if training_method == "Q-Learning":
    alpha = st.sidebar.slider("Learning rate (α)", 0.01, 1.0, 0.8, help="Controls how much the agent updates its Q-values")
    epsilon = st.sidebar.slider("Exploration rate (ε)", 0.01, 1.0, 0.1, help="Probability of taking a random action")
    episodes = st.sidebar.slider("Training Episodes", 100, 10000, 1000)
else:  # Value Iteration
    theta = st.sidebar.slider("Convergence threshold (θ)", 0.0001, 0.1, 0.001, format="%.4f", help="Threshold for determining convergence")
    max_iterations = st.sidebar.slider("Maximum Iterations", 10, 1000, 100, help="Maximum number of iterations for value iteration")

# Import Value Iteration
from value_iteration import ValueIteration

# Initialize environment and agent
if not st.session_state.initialized or st.session_state.Q is None:
    env = gym.make("FrozenLake-v1", is_slippery=st.session_state.is_slippery)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))
    st.session_state.Q = np.zeros((n_states, n_actions))
    st.session_state.state = env.reset()[0]
    st.session_state.done = False
    st.session_state.path = [st.session_state.state]
    st.session_state.initialized = True
    # Store environment in session state to maintain consistency
    st.session_state.env = env
    
    # Initialize Value Iteration agent if not exists
    if 'value_iteration_agent' not in st.session_state:
        st.session_state.value_iteration_agent = ValueIteration(is_slippery=st.session_state.is_slippery)
        st.session_state.V = None
        st.session_state.policy = None
else:
    # Use the stored environment if available, otherwise create a new one
    if hasattr(st.session_state, 'env'):
        env = st.session_state.env
    else:
        env = gym.make("FrozenLake-v1", is_slippery=st.session_state.is_slippery)
        st.session_state.env = env
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))
    
    # Update Value Iteration agent if slippery setting changed
    if 'value_iteration_agent' not in st.session_state or st.session_state.value_iteration_agent.env.spec.kwargs['is_slippery'] != st.session_state.is_slippery:
        st.session_state.value_iteration_agent = ValueIteration(is_slippery=st.session_state.is_slippery)

# Set title based on selected method
if training_method == "Q-Learning":
    st.title("🤖 Q-Learning Visualized - FrozenLake")
else:
    st.title("🤖 Value Iteration Visualized - FrozenLake")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Training & Visualization", "🎮 Interactive Mode", "📘 About RL"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Training Progress")
        
        # Training controls
        train_col1, train_col2 = st.columns(2)
        with train_col1:
            train_button = st.button("🚀 Train Agent", use_container_width=True)
        with train_col2:
            reset_button = st.button("🔄 Reset Training", use_container_width=True)
        
        if reset_button:
            st.session_state.Q = np.zeros((n_states, n_actions))
            st.session_state.rewards_history = []
            st.session_state.success_rate_history = []
            st.session_state.episode_count = 0
            st.session_state.training_complete = False
            st.rerun()
        
        if train_button or (not st.session_state.training_complete and len(st.session_state.rewards_history) == 0):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if training_method == "Q-Learning":
                # Q-Learning Training loop
                batch_size = min(100, episodes)  # Process in batches for better UI responsiveness
                num_batches = episodes // batch_size
                
                rewards_per_episode = []
                success_count = 0
                
                for batch in range(num_batches):
                    batch_successes = 0
                    batch_rewards = []
                    
                    for _ in range(batch_size):
                        state = env.reset()[0]
                        done = False
                        total_reward = 0
                        steps = 0
                        max_steps = 100  # Prevent infinite loops
                        
                        while not done and steps < max_steps:
                            # Epsilon-greedy action selection
                            if np.random.rand() < epsilon:
                                action = env.action_space.sample()
                            else:
                                action = np.argmax(st.session_state.Q[state])
                            
                            next_state, reward, done, _, _ = env.step(action)
                            
                            # Q-learning update
                            st.session_state.Q[state, action] += alpha * (
                                reward + gamma * np.max(st.session_state.Q[next_state]) - st.session_state.Q[state, action]
                            )
                            
                            state = next_state
                            total_reward += reward
                            steps += 1
                        
                        rewards_per_episode.append(total_reward)
                        batch_rewards.append(total_reward)
                        if total_reward > 0:  # Agent reached the goal
                            batch_successes += 1
                            success_count += 1
                    
                    # Update progress
                    progress = (batch + 1) / num_batches
                    progress_bar.progress(progress)
                    
                    # Update metrics
                    st.session_state.episode_count += batch_size
                    st.session_state.rewards_history.extend(batch_rewards)
                    current_success_rate = batch_successes / batch_size
                    st.session_state.success_rate_history.append(current_success_rate)
                    
                    status_text.text(f"Training: {batch+1}/{num_batches} batches completed | Success rate: {current_success_rate:.2f}")
                    time.sleep(0.01)  # Small delay for UI updates
                
                st.session_state.training_complete = True
                progress_bar.progress(1.0)
                status_text.text(f"Training complete! Overall success rate: {success_count/episodes:.2f}")
            
            else:  # Value Iteration
                # Value Iteration training
                def update_callback(iteration, delta, V, policy):
                    progress = min(1.0, (iteration + 1) / max_iterations)
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration: {iteration+1}/{max_iterations} | Delta: {delta:.6f}")
                    time.sleep(0.01)  # Small delay for UI updates
                
                # Run Value Iteration
                V, policy, iterations, converged = st.session_state.value_iteration_agent.train(
                    gamma=gamma, 
                    theta=theta, 
                    max_iterations=max_iterations,
                    callback=update_callback
                )
                
                # Store results in session state
                st.session_state.V = V
                st.session_state.policy = policy
                st.session_state.training_complete = True
                
                # Update progress bar to completion
                progress_bar.progress(1.0)
                
                if converged:
                    status_text.text(f"Training complete! Converged after {iterations} iterations.")
                else:
                    status_text.text(f"Training complete! Reached maximum iterations ({max_iterations}).")
                
                # For consistency with Q-learning interface
                st.session_state.episode_count = iterations
                
                # Get convergence history for plotting
                training_history = st.session_state.value_iteration_agent.get_training_history()
                st.session_state.rewards_history = [0] * len(training_history['convergence_history'])  # Placeholder
                st.session_state.success_rate_history = [1.0 - delta for delta in training_history['convergence_history']]  # Use 1-delta as a proxy for success rate
        
        # Display metrics
        if len(st.session_state.rewards_history) > 0:
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Episodes Trained", st.session_state.episode_count)
            with metrics_col2:
                if len(st.session_state.success_rate_history) > 0:
                    latest_success_rate = st.session_state.success_rate_history[-1]
                    st.metric("Latest Success Rate", f"{latest_success_rate:.2f}")
            
            # Plot training metrics
            st.subheader("Learning Curves")
            chart_tab1, chart_tab2 = st.tabs(["Success Rate", "Rewards"])
            
            with chart_tab1:
                if len(st.session_state.success_rate_history) > 0:
                    success_df = pd.DataFrame({
                        'Batch': range(1, len(st.session_state.success_rate_history) + 1),
                        'Success Rate': st.session_state.success_rate_history
                    })
                    st.line_chart(success_df.set_index('Batch'))
            
            with chart_tab2:
                if len(st.session_state.rewards_history) > 0:
                    # Use a rolling window to smooth the rewards curve
                    window_size = min(50, len(st.session_state.rewards_history))
                    rewards_smoothed = np.convolve(st.session_state.rewards_history, 
                                                np.ones(window_size)/window_size, mode='valid')
                    rewards_df = pd.DataFrame({
                        'Episode': range(1, len(rewards_smoothed) + 1),
                        'Reward (smoothed)': rewards_smoothed
                    })
                    st.line_chart(rewards_df.set_index('Episode'))
    
    with col2:
        if training_method == "Q-Learning":
            st.subheader("Q-Value Visualization")
            
            if st.session_state.Q is not None:
                # Create a heatmap of the maximum Q-value for each state
                q_max = np.max(st.session_state.Q, axis=1).reshape(grid_size, grid_size)
                
                # Create a custom colormap from blue to red
                cmap = LinearSegmentedColormap.from_list(
                    "custom_cmap", [(0, "#2171b5"), (0.5, "#6baed6"), (0.75, "#fcae91"), (1, "#cb181d")]
                )
                
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(q_max, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, ax=ax, cbar=True)
                ax.set_title("Maximum Q-value per State")
                st.pyplot(fig)
                
                # Display the policy (best action for each state)
                st.subheader("Learned Policy")
                policy = np.argmax(st.session_state.Q, axis=1).reshape(grid_size, grid_size)
                
                # Convert action indices to arrows
                action_symbols = ['←', '↓', '→', '↑']
                policy_symbols = np.array([[action_symbols[a] for a in row] for row in policy])
                
                # Mark holes and goal
                holes = [5, 7, 11, 12]  # Hole positions in 4x4 grid
                goal = grid_size * grid_size - 1  # Goal position
                
                for hole in holes:
                    row, col = hole // grid_size, hole % grid_size
                    policy_symbols[row, col] = '🕳️'
                
                goal_row, goal_col = goal // grid_size, goal % grid_size
                policy_symbols[goal_row, goal_col] = '🏁'
                
                # Create a DataFrame for better display
                policy_df = pd.DataFrame(policy_symbols)
                
                # Style the DataFrame
                def highlight_policy(val):
                    color = '#ffffff'
                    if val == '🕳️':
                        color = '#ff7f7f'  # Light red for holes
                    elif val == '🏁':
                        color = '#90ee90'  # Light green for goal
                    return f'background-color: {color}; font-size: 20px; text-align: center;'
                
                styled_policy = policy_df.style.map(highlight_policy)
                st.dataframe(styled_policy, use_container_width=True, height=200)
        
        else:  # Value Iteration
            st.subheader("Value Function Visualization")
            
            if st.session_state.V is not None:
                # Create a heatmap of the value function
                v_grid = st.session_state.V.reshape(grid_size, grid_size)
                
                # Create a custom colormap from blue to red
                cmap = LinearSegmentedColormap.from_list(
                    "custom_cmap", [(0, "#2171b5"), (0.5, "#6baed6"), (0.75, "#fcae91"), (1, "#cb181d")]
                )
                
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(v_grid, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, ax=ax, cbar=True)
                ax.set_title("State Value Function")
                st.pyplot(fig)
                
                # Display the policy
                st.subheader("Optimal Policy")
                
                # Get policy symbols from the value iteration agent
                policy_symbols = st.session_state.value_iteration_agent.get_policy_table()
                
                # Create a DataFrame for better display
                policy_df = pd.DataFrame(policy_symbols)
                
                # Style the DataFrame
                def highlight_policy(val):
                    color = '#ffffff'
                    if val == '🕳️':
                        color = '#ff7f7f'  # Light red for holes
                    elif val == '🏁':
                        color = '#90ee90'  # Light green for goal
                    return f'background-color: {color}; font-size: 20px; text-align: center;'
                
                styled_policy = policy_df.style.map(highlight_policy)
                st.dataframe(styled_policy, use_container_width=True, height=200)

with tab2:
    st.subheader("Interactive Agent Navigation")
    
    # Reset interactive environment
    if st.button("🔄 Reset Environment"):
        # Reset the environment and update session state
        st.session_state.state = st.session_state.env.reset()[0]
        st.session_state.env.reset_called = True  # Mark that reset has been called
        st.session_state.done = False
        st.session_state.path = [st.session_state.state]
    
    # Display current state
    render_grid(st.session_state.state)
    
    # Navigation controls
    st.markdown("### Control the Agent")
    
    # Two options: manual control or follow policy
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        # Manual controls
        st.markdown("#### Manual Control")
        manual_col1, manual_col2, manual_col3 = st.columns([1, 1, 1])
        with manual_col1:
            st.write("")
            up_button = st.button("↑", use_container_width=True)
        with manual_col2:
            left_button = st.button("←", use_container_width=True)
            down_button = st.button("↓", use_container_width=True)
        with manual_col3:
            right_button = st.button("→", use_container_width=True)
        
        # Process manual actions
        action = None
        if up_button:
            action = 3  # Up
        elif down_button:
            action = 1  # Down
        elif left_button:
            action = 0  # Left
        elif right_button:
            action = 2  # Right
        
        if action is not None and not st.session_state.done:
            # Ensure environment is reset before stepping
            if not hasattr(st.session_state.env, 'reset_called') or not st.session_state.env.reset_called:
                st.session_state.env.reset()
                st.session_state.env.reset_called = True
            next_state, reward, done, _, _ = st.session_state.env.step(action)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.path.append(next_state)
            st.rerun()
    
    with control_col2:
        # Policy-based navigation
        st.markdown("#### Follow Policy")
        if st.button("Take Best Action", use_container_width=True) and not st.session_state.done:
            state = st.session_state.state
            
            # Get action based on selected method
            if training_method == "Q-Learning":
                action = np.argmax(st.session_state.Q[state])
            else:  # Value Iteration
                if st.session_state.policy is not None:
                    action = st.session_state.policy[state]
                else:
                    action = env.action_space.sample()  # Random action if policy not trained
            
            # Ensure environment is reset before stepping
            if not hasattr(st.session_state.env, 'reset_called') or not st.session_state.env.reset_called:
                st.session_state.env.reset()
                st.session_state.env.reset_called = True
            next_state, reward, done, _, _ = st.session_state.env.step(action)
            st.session_state.state = next_state
            st.session_state.done = done
            st.session_state.path.append(next_state)
            st.rerun()
    
    # Display result if done
    if st.session_state.done:
        if st.session_state.state == grid_size * grid_size - 1:
            st.success("🎉 Agent reached the goal!")
        else:
            st.error("💥 Agent fell into a hole!")
    
    # Display path taken
    st.markdown("### Path Taken")
    path_str = " → ".join([str(s) for s in st.session_state.path])
    st.code(path_str)

with tab3:
    st.subheader("About Reinforcement Learning")
    
    st.markdown("""
    ### What is Reinforcement Learning?
    
    Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards. Unlike supervised learning, RL doesn't require labeled data but learns through trial and error.
    
    ### Training Methods
    
    #### Q-Learning Algorithm
    
    Q-Learning is a model-free reinforcement learning algorithm that learns the value of an action in a particular state. It creates a Q-table where:
    
    - Rows represent states
    - Columns represent possible actions
    - Values represent the expected future rewards of taking that action in that state
    
    The Q-value update formula is:
    
    Q(s,a) ← Q(s,a) + α[r + γ·max<sub>a'</sub>Q(s',a') - Q(s,a)]
    
    Where:
    - α (alpha) is the learning rate
    - γ (gamma) is the discount factor
    - r is the reward
    - s is the current state
    - a is the action taken
    - s' is the next state
    
    #### Value Iteration Algorithm
    
    Value Iteration is a model-based algorithm that computes the optimal state-value function by iteratively improving the value estimates. The algorithm works as follows:
    
    1. Initialize the value function V(s) for all states
    2. Repeatedly update V(s) using the Bellman optimality equation:
       V(s) ← max<sub>a</sub> Σ<sub>s'</sub> P(s'|s,a)[R(s,a,s') + γV(s')]
    3. Continue until the value function converges (changes less than threshold θ)
    4. Extract the optimal policy by selecting actions that maximize expected value
    
    Where:
    - γ (gamma) is the discount factor
    - P(s'|s,a) is the transition probability
    - R(s,a,s') is the reward function
    - θ (theta) is the convergence threshold
    
    ### The Frozen Lake Environment
    
    In this environment, the agent navigates a frozen lake from the starting point (top-left) to the goal (bottom-right) without falling into holes. The lake surface is slippery (when enabled), making the agent's movements uncertain.
    
    - 🤖 Agent: The learning algorithm
    - 🏁 Goal: The target destination
    - 🕳️ Holes: Areas to avoid
    - ⬜ Frozen surface: Safe to walk on
    
    ### Parameters
    
    - **Learning rate (α)**: Controls how quickly the agent updates its knowledge
    - **Discount factor (γ)**: Determines the importance of future rewards
    - **Exploration rate (ε)**: Controls the exploration-exploitation trade-off
    
    Experiment with these parameters to see how they affect the agent's learning process!
    """)
    
    st.info("This application demonstrates the power of reinforcement learning through interactive visualization. Use the tabs to train the agent, test its performance, and learn about the underlying concepts.")

# Step-by-step animation section
# This section is now handled by the render_grid function defined at the top of the file

# Step-by-step animation section
# This section is now handled by the render_grid function defined at the top of the file

# Step-by-step animation
if st.button("➡️ Next Step"):
    if not st.session_state.done:
        state = st.session_state.state
        
        # Get action based on selected method
        if training_method == "Q-Learning":
            action = np.argmax(st.session_state.Q[state])
        else:  # Value Iteration
            if st.session_state.policy is not None:
                action = st.session_state.policy[state]
            else:
                action = env.action_space.sample()  # Random action if policy not trained
        
        # Ensure environment is reset before stepping
        if not hasattr(st.session_state.env, 'reset_called') or not st.session_state.env.reset_called:
            st.session_state.env.reset()
            st.session_state.env.reset_called = True
        next_state, reward, done, _, _ = st.session_state.env.step(action)
        st.session_state.state = next_state
        st.session_state.done = done
        st.session_state.path.append(next_state)
        st.rerun()
    else:
        if st.session_state.state == (int(np.sqrt(st.session_state.env.observation_space.n)) ** 2) - 1:
            st.success("🎉 Agent reached the goal!")
        else:
            st.error("💥 Agent fell into a hole!")