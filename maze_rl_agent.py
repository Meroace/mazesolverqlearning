import numpy as np
import random
import time
import matplotlib.pyplot as plt
import csv

# Constants
GRID_SIZE = 10  # 10x10 grid
BLOCK_SIZE = 5  # Size for visualization (doesn't affect non-GUI)
FPS = 5  # Frames per second (this doesn't matter in CLI mode)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREY = (169, 169, 169)

# Define the maze (10x10 grid)
# 0 = open space, -1 = obstacle, 10 = goal
maze = np.zeros((GRID_SIZE, GRID_SIZE))
maze[1:3, 1:3] = -1  # Obstacles
maze[4:5, 4:5] = -1
maze[6:7, 6:7] = -1
maze[9, 9] = 10  # Goal

# Q-Learning parameters
gamma = 0.9  # Discount factor
alpha = 0.8  # Learning rate
epsilon = 0.2  # Exploration rate
num_episodes = 1000  # Number of training episodes

# Initialize Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  # 4 actions (up, down, left, right)

# Actions: 0 = up, 1 = down, 2 = left, 3 = right
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Define the start and multiple goals
start = (0, 0)
goals = [(9, 9)]

# Data to track during training
reward_progress = []  # List to store cumulative reward for each episode
success_count = 0  # Track how many times the agent reaches the goal
convergence_episode = None  # Episode where convergence is reached

# Function to choose the best action (epsilon-greedy)
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2, 3])  # Random action (explore)
    else:
        return np.argmax(q_table[state[0], state[1]])  # Best action (exploit)

# Function to move the agent
def move(state, action):
    new_state = (state[0] + actions[action][0], state[1] + actions[action][1])

    # Check if the new state is within bounds and not an obstacle
    if (0 <= new_state[0] < GRID_SIZE and 0 <= new_state[1] < GRID_SIZE and maze[new_state[0], new_state[1]] != -1):
        return new_state
    else:
        return state  # Stay in the current position if move is invalid

# Function to add moving obstacles
def move_obstacles():
    # Move obstacles randomly every 100 episodes
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if maze[i, j] == -1:
                # Move each obstacle by a random amount within the grid
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                new_i, new_j = i + direction[0], j + direction[1]
                if 0 <= new_i < GRID_SIZE and 0 <= new_j < GRID_SIZE and maze[new_i, new_j] != 10 and maze[new_i, new_j] != -1:
                    maze[i, j] = 0
                    maze[new_i, new_j] = -1
    return maze

# Function to log maze state after obstacle movement
def log_maze_state(episode):
    if episode % 100 == 0:
        print(f"\nMaze state at episode {episode}:")
        print(maze)
        # Optionally save to a file for later use
        np.savetxt(f"maze_state_episode_{episode}.txt", maze, fmt='%d')

# Function to track convergence (success rate)
def track_convergence():
    global convergence_episode
    success_threshold = 0.95  # 95% success rate over 100 episodes
    consecutive_success = 0
    episode = 0

    while episode < num_episodes:
        state = start
        success = False
        
        while state not in goals:
            action = choose_action(state)
            next_state = move(state, action)
            if next_state in goals:
                success = True
                break
            state = next_state
        
        if success:
            consecutive_success += 1
        else:
            consecutive_success = 0
        
        if consecutive_success >= success_threshold * 100:
            convergence_episode = episode
            print(f"Convergence reached at episode {episode}")
            break
        
        episode += 1

# Q-Learning Algorithm
def train_q_learning():
    global success_count
    episode = 0
    while episode < num_episodes:
        state = start
        total_reward = 0
        
        while state not in goals:
            action = choose_action(state)
            next_state = move(state, action)
            reward = -1  # Negative reward for each step
            if next_state in goals:
                reward = 10  # Goal reward
                success_count += 1  # Increment success count when goal is reached
            
            # Update Q-value
            q_table[state[0], state[1], action] += alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])
            
            state = next_state
            total_reward += reward
        
        # Store the cumulative reward for the current episode
        reward_progress.append(total_reward)

        # Change maze periodically
        if episode % 100 == 0:
            maze = move_obstacles()  # Move obstacles
            log_maze_state(episode)  # Log maze state
        
        episode += 1

# Function to simulate the agent after training
def simulate_agent():
    state = start
    path = [state]
    
    while state not in goals:
        print("\nCurrent state:")
        draw_maze(state)
        action = np.argmax(q_table[state[0], state[1]])  # Best action
        state = move(state, action)
        path.append(state)
        
        time.sleep(0.5)  # Slow down the movement for visualization
    print("\nGoal reached!")

# Visualization of the maze and agent in the terminal
def draw_maze(agent_pos):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if maze[i, j] == -1:
                print("X", end=" ")  # Wall
            elif maze[i, j] == 10:
                print("G", end=" ")  # Goal
            elif (i, j) == agent_pos:
                print("A", end=" ")  # Agent
            else:
                print(".", end=" ")  # Open space
        print("")  # New line for next row

# Plot the learning curve
def plot_learning_curve():
    plt.plot(reward_progress)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Curve')
    plt.show()

# Save the reward progress to a CSV file
def save_reward_progress():
    with open("reward_progress.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Cumulative Reward"])
        for i, reward in enumerate(reward_progress):
            writer.writerow([i, reward])

# Run the Q-learning agent for training
train_q_learning()

# Plot the learning curve after training
plot_learning_curve()

# Simulate the trained agent
simulate_agent()

# Save the reward progress
save_reward_progress()

# Calculate and display success rate
success_rate = success_count / num_episodes
print(f"Success rate: {success_rate * 100:.2f}%")

# Track convergence
track_convergence()

