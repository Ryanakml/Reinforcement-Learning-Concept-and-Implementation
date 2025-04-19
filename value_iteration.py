import numpy as np
import gym
import pandas as pd
import time

class ValueIteration:
    def __init__(self, is_slippery=False):
        """
        Initialize the Value Iteration algorithm for the FrozenLake environment.
        
        Args:
            is_slippery (bool): Whether the environment is slippery or not.
        """
        self.env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.grid_size = int(np.sqrt(self.n_states))
        
        # Initialize value function and policy
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=int)
        
        # For tracking training progress
        self.v_history = []
        self.policy_history = []
        self.convergence_history = []
        
    def train(self, gamma=0.95, theta=0.0001, max_iterations=1000, callback=None):
        """
        Train the agent using Value Iteration algorithm.
        
        Args:
            gamma (float): Discount factor.
            theta (float): Convergence threshold.
            max_iterations (int): Maximum number of iterations.
            callback (function): Optional callback function to track progress.
            
        Returns:
            tuple: (V, policy, iterations, converged)
        """
        # Initialize value function
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        iterations = 0
        converged = False
        
        # Store initial state for history
        self.v_history.append(V.copy())
        self.policy_history.append(policy.copy())
        self.convergence_history.append(1.0)  # Start with max difference
        
        for i in range(max_iterations):
            delta = 0
            # Update each state
            for s in range(self.n_states):
                v = V[s]
                
                # Look at the possible next actions
                action_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    # Look at the possible next states
                    for prob, next_s, reward, done in self.env.P[s][a]:
                        # Calculate the expected value
                        action_values[a] += prob * (reward + gamma * V[next_s])
                
                # Select the best action based on the highest expected value
                best_action = np.argmax(action_values)
                best_value = action_values[best_action]
                
                # Calculate change in value
                delta = max(delta, abs(v - best_value))
                
                # Update the value function
                V[s] = best_value
                
                # Update the policy
                policy[s] = best_action
            
            # Store history for visualization
            self.v_history.append(V.copy())
            self.policy_history.append(policy.copy())
            self.convergence_history.append(delta)
            
            # Check for convergence
            if delta < theta:
                converged = True
                break
                
            iterations += 1
            
            # Call the callback function if provided
            if callback is not None:
                callback(i, delta, V, policy)
        
        # Store the final value function and policy
        self.V = V
        self.policy = policy
        
        return V, policy, iterations, converged
    
    def get_action(self, state):
        """
        Get the best action for a given state based on the learned policy.
        
        Args:
            state (int): The current state.
            
        Returns:
            int: The best action to take.
        """
        return self.policy[state]
    
    def reset(self):
        """
        Reset the environment and return the initial state.
        
        Returns:
            int: The initial state.
        """
        return self.env.reset()[0]
    
    def step(self, state, action):
        """
        Take a step in the environment.
        
        Args:
            state (int): The current state.
            action (int): The action to take.
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        next_state, reward, done, _, info = self.env.step(action)
        return next_state, reward, done, info
    
    def get_v_table(self):
        """
        Get the value function table.
        
        Returns:
            numpy.ndarray: The value function reshaped to match the grid.
        """
        return self.V.reshape(self.grid_size, self.grid_size)
    
    def get_policy_table(self):
        """
        Get the policy table with action symbols.
        
        Returns:
            numpy.ndarray: The policy with action symbols.
        """
        action_symbols = ['←', '↓', '→', '↑']
        policy_grid = self.policy.reshape(self.grid_size, self.grid_size)
        policy_symbols = np.array([[action_symbols[a] for a in row] for row in policy_grid])
        
        # Mark holes and goal
        holes = [5, 7, 11, 12]  # Hole positions in 4x4 grid
        goal = self.grid_size * self.grid_size - 1  # Goal position
        
        for hole in holes:
            row, col = hole // self.grid_size, hole % self.grid_size
            policy_symbols[row, col] = '🕳️'
        
        goal_row, goal_col = goal // self.grid_size, goal % self.grid_size
        policy_symbols[goal_row, goal_col] = '🏁'
        
        return policy_symbols
    
    def get_training_history(self):
        """
        Get the training history.
        
        Returns:
            dict: The training history.
        """
        return {
            'v_history': self.v_history,
            'policy_history': self.policy_history,
            'convergence_history': self.convergence_history
        }
    
    def save_training_data(self, file_path='assets/value_iteration.csv'):
        """
        Save the training data to a CSV file.
        
        Args:
            file_path (str): The file path to save the data.
        """
        # Save the value function
        v_df = pd.DataFrame(self.V.reshape(1, -1))
        v_df.to_csv('assets/v_values.csv', index=False)
        
        # Save the convergence history
        conv_df = pd.DataFrame({
            'iteration': range(len(self.convergence_history)),
            'delta': self.convergence_history
        })
        conv_df.to_csv(file_path, index=False)
        
        # Save state transitions for visualization
        transitions = []
        for s in range(self.n_states):
            a = self.policy[s]
            for prob, next_s, reward, done in self.env.P[s][a]:
                if prob > 0:
                    transitions.append({
                        'state': s,
                        'action': a,
                        'next_state': next_s,
                        'probability': prob,
                        'reward': reward,
                        'done': done
                    })
        
        trans_df = pd.DataFrame(transitions)
        trans_df.to_csv('assets/state_transitions.csv', index=False)

# Example usage
def example_usage():
    # Initialize the Value Iteration agent
    vi = ValueIteration(is_slippery=True)
    
    # Define a callback function to track progress
    def callback(iteration, delta, V, policy):
        print(f"Iteration {iteration}: delta = {delta}")
    
    # Train the agent
    V, policy, iterations, converged = vi.train(
        gamma=0.95,
        theta=0.0001,
        max_iterations=1000,
        callback=callback
    )
    
    print(f"\nTraining completed after {iterations} iterations")
    print(f"Converged: {converged}")
    
    # Print the value function
    print("\nValue Function:")
    print(vi.get_v_table())
    
    # Print the policy
    print("\nPolicy:")
    print(vi.get_policy_table())
    
    # Save the training data
    vi.save_training_data()

if __name__ == "__main__":
    example_usage()