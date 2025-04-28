# Reduce number of episodes and increase FPS
num_episodes = 100  # Reduced number of episodes for faster execution
FPS = 30  # Increased FPS to speed up the simulation

# Function to train the Q-learning agent
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
        
        reward_progress.append(total_reward)

        # Periodically update maze and log the state
        if episode % 100 == 0:
            maze = move_obstacles()  # Move obstacles
            log_maze_state(episode)  # Log maze state
        
        # Draw maze during training (optional, for visualization)
        # draw_maze(state)  # Comment this out for faster execution
        
        # Logging progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}")
        
        # Check for Pygame quit event (to close the window properly)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        pygame.display.update()  # Update the Pygame display
        clock.tick(FPS)  # Control the speed of the loop

        episode += 1

