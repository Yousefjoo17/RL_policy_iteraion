import numpy as np
import random
import gym
from gym import spaces
import pygame
import imageio
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# =====================
# GridWorld Gym Environment with Stochastic Movement
# =====================
class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, num_barriers=2, r_goal=10, r_barrier=-10, seed_nr=0):
        super(GridWorldEnv, self).__init__()
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        self.n = n
        self.action_space = spaces.Discrete(4)  # left, up, right, down
        self.observation_space = spaces.Box(low=0, high=n-1, shape=(2,), dtype=int)

        self.actions = [0, 1, 2, 3]  # left, up, right, down
        self.rewards = (-1) * np.ones((n, n))
        self.agent_pos = (random.randrange(n), random.randrange(n))

        possible_goal_positions = [(i, j) for i in range(n) for j in range(n) if (i, j) != self.agent_pos]
        self.goal = random.choice(possible_goal_positions)

        self.rewards[self.goal[0], self.goal[1]] = r_goal
        # Place barriers explicitly
        possible_barrier_positions = [(i, j) for i in range(n) for j in range(n)
                                      if (i, j) != self.goal and (i, j) != self.agent_pos]
        barrier_positions = random.sample(possible_barrier_positions, num_barriers)
        for (i, j) in barrier_positions:
            self.rewards[i, j] = r_barrier

        self.cell_size = 75
        self.screen = None
        self.clock = None

    def reset(self):
        self.agent_pos = (random.randrange(self.n), random.randrange(self.n))
        return np.array(self.agent_pos, dtype=int)

    def step(self, action):
        if action == 0:  # left
            probabilities = [0.7, 0.15, 0.0, 0.15]  # left, up, right, down
        elif action == 1:  # up
            probabilities = [0.15, 0.7, 0.15, 0.0]
        elif action == 2:  # right
            probabilities = [0.0, 0.15, 0.7, 0.15]
        elif action == 3:  # down
            probabilities = [0.15, 0.0, 0.15, 0.7]

        actual_action = np.random.choice(4, p=probabilities)

        x, y = self.agent_pos
        n = self.n
        if actual_action == 0:
            y = max(0, y - 1)
        elif actual_action == 1:
            x = max(0, x - 1)
        elif actual_action == 2:
            y = min(n - 1, y + 1)
        elif actual_action == 3:
            x = min(n - 1, x + 1)

        self.agent_pos = (x, y)
        reward = self.rewards[x, y]

        done = (x, y) == self.goal

        return np.array(self.agent_pos, dtype=int), reward, done, {}

    def render(self, save_frames=False, frame_list=None):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.n * self.cell_size, self.n * self.cell_size))
            self.clock = pygame.time.Clock()

        colors = {
            "empty": (255, 255, 255),
            "barrier": (255, 0, 0),
            "goal": (0, 255, 0),
            "agent": (0, 0, 255)
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for i in range(self.n):
            for j in range(self.n):
                cell_color = colors["empty"]
                if self.rewards[i, j] < -1:
                    cell_color = colors["barrier"]
                if (i, j) == self.goal:
                    cell_color = colors["goal"]

                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                )
                pygame.draw.rect(
                    self.screen, (0, 0, 0),
                    pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    1
                )

        ax, ay = self.agent_pos
        pygame.draw.circle(
            self.screen,
            colors["agent"],
            (ay * self.cell_size + self.cell_size // 2, ax * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )

        pygame.display.flip()
        self.clock.tick(4)

        if save_frames:
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = np.transpose(frame, (1, 0, 2))
            frame_list.append(frame)

    def close(self):
        if self.screen:
            pygame.quit()


# =====================
# Policy Iteration Logic
# =====================
def policy_iteration(env, v0_val=0, gamma=0.9, theta=1e-3):
    n = env.n
    v = v0_val * np.ones((n, n))
    pi = 1 / 4 * np.ones((n, n, 4))
    policy_stable = False
    iteration = 1
    start_time = time.time()
    
    while not policy_stable:
        print(f"Iteration number {iteration}")
        policy_eval_iterations = policy_evaluation(env, v, pi, gamma, theta)  # updates V
        print(f"Took {policy_eval_iterations} iterations to converge for policy_eval")
        policy_stable = policy_improvement(env, v, pi, gamma)  # updates Pi
        iteration += 1
    
    end_time = time.time()
    print(f"Converged in {iteration-1} policy iterations")
    print(f"Time taken: {end_time - start_time:.6f} seconds")
    
    return pi, v


def policy_evaluation(env, v, pi, gamma, theta):
    n = env.n
    delta = float("inf")
    iterations = 0
    
    while delta > theta:
        iterations += 1
        delta = 0
        v_old = v.copy()
        for x in range(n):
            for y in range(n):
                if (x, y) == env.goal:
                    continue
                total = 0
                for a in env.actions:
                    expected_value = 0
                    for actual_action, prob in get_transition_probabilities(a):
                        s_prime_x, s_prime_y = get_next_state(x, y, actual_action, n)
                        expected_value += prob * (env.rewards[s_prime_x, s_prime_y] + gamma * v_old[s_prime_x, s_prime_y])
                    total += pi[x, y, a] * expected_value
                delta = max(delta, abs(v[x, y] - total))
                v[x, y] = total
                
    return iterations


def policy_improvement(env, v, pi, gamma):
    policy_stable = True
    n = env.n
    for x in range(n):
        for y in range(n):
            old_pi = pi[x, y, :].copy()
            q_values = np.zeros(4)
            for a in env.actions:
                q_value = 0
                for actual_action, prob in get_transition_probabilities(a):
                    s_prime_x, s_prime_y = get_next_state(x, y, actual_action, n)
                    q_value += prob * (env.rewards[s_prime_x, s_prime_y] + gamma * v[s_prime_x, s_prime_y])
                q_values[a] = q_value

            best_actions = np.argwhere(q_values == np.max(q_values)).flatten()
            pi[x, y, :] = 0
            for a in best_actions:
                pi[x, y, a] = 1 / len(best_actions)

            if not np.array_equal(old_pi, pi[x, y, :]):
                policy_stable = False
    return policy_stable


def get_transition_probabilities(intended_action):
    if intended_action == 0:  # left
        return [(0, 0.7), (1, 0.15), (3, 0.15)]
    elif intended_action == 1:  # up
        return [(1, 0.7), (0, 0.15), (2, 0.15)]
    elif intended_action == 2:  # right
        return [(2, 0.7), (1, 0.15), (3, 0.15)]
    elif intended_action == 3:  # down
        return [(3, 0.7), (0, 0.15), (2, 0.15)]


def get_next_state(x, y, a, n):
    if a == 0:  # left
        return x, max(0, y - 1)
    elif a == 1:  # up
        return max(0, x - 1), y
    elif a == 2:  # right
        return x, min(n - 1, y + 1)
    else:  # down
        return min(n - 1, x + 1), y


# =====================
# Plotting Functions (Save only)
# =====================
def save_value_function(env, value_function, filename="value_function.png"):
    """Save the value function as a heatmap with explicit cell types"""
    plt.figure(figsize=(10, 8))
    
    # Create a custom colormap that handles special cells
    masked_values = value_function.copy()
    
    # Create background colors for special cells
    for i in range(env.n):
        for j in range(env.n):
            if env.rewards[i, j] < -1:  # Barrier
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='red', alpha=0.3))
                plt.text(j, i, 'BARRIER', ha='center', va='center', fontweight='bold', fontsize=8)
            elif (i, j) == env.goal:  # Goal
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='green', alpha=0.3))
                plt.text(j, i, 'GOAL', ha='center', va='center', fontweight='bold', fontsize=8)
            else:  # Regular cell - show value
                plt.text(j, i, f'{value_function[i, j]:.1f}', 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Plot the heatmap for regular cells
    im = plt.imshow(value_function, cmap='viridis', alpha=0.6)
    plt.colorbar(im, label='Value')
    
    plt.title('Value Function with Cell Types')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(range(env.n))
    plt.yticks(range(env.n))
    plt.tight_layout()
    
    # Save instead of showing
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Value function saved as {filename}")

def save_policy(env, policy, filename="policy.png"):
    """Save the policy using arrows with explicit cell types"""
    plt.figure(figsize=(10, 8))
    
    # Create the grid background
    for i in range(env.n):
        for j in range(env.n):
            # Set background color based on cell type
            if env.rewards[i, j] < -1:  # Barrier
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='red', alpha=0.5))
                plt.text(j, i, 'BARRIER', ha='center', va='center', fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.8))
            elif (i, j) == env.goal:  # Goal
                plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='green', alpha=0.5))
                plt.text(j, i, 'GOAL', ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.8))
            else:  # Regular cell - show policy arrows
                # Get the best action(s)
                best_actions = np.where(policy[i, j, :] > 0)[0]
                
                # Draw arrows for each possible action in the policy
                for action in best_actions:
                    dx, dy = 0, 0
                    if action == 0:  # left
                        dx = -0.3
                        arrow_color = 'blue'
                    elif action == 1:  # up
                        dy = -0.3
                        arrow_color = 'blue'
                    elif action == 2:  # right
                        dx = 0.3
                        arrow_color = 'blue'
                    elif action == 3:  # down
                        dy = 0.3
                        arrow_color = 'blue'
                    
                    plt.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1, 
                             fc=arrow_color, ec=arrow_color, length_includes_head=True)
                
                # Show the cell coordinates for regular cells
                plt.text(j, i + 0.25, f'({i},{j})', ha='center', va='center', 
                        fontsize=6, color='gray')
    
    plt.xlim(-0.5, env.n-0.5)
    plt.ylim(env.n-0.5, -0.5)
    plt.title('Optimal Policy with Cell Types')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(range(env.n))
    plt.yticks(range(env.n))
    plt.tight_layout()
    
    # Save instead of showing
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Policy plot saved as {filename}")


# =====================
# Simulate and Record Video
# =====================
def simulate_and_record(env, pi, filename="gridworld_policy_iteration_stochastic.mp4"):
    frames = []
    state = env.reset()
    done = False
    env.render(save_frames=True, frame_list=frames)

    while not done:
        x, y = state
        action = np.argmax(pi[x, y, :])
        state, _, done, _ = env.step(action)
        env.render(save_frames=True, frame_list=frames)

    env.close()
    imageio.mimsave(filename, frames, fps=4)
    print(f"🎥 Simulation video saved as {filename}")


# =====================
# Main
# =====================
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    
    env = GridWorldEnv(n=5, num_barriers=2, r_goal=5, r_barrier=-5, seed_nr=1)
    start_pos = env.agent_pos
    
    print("=" * 50)
    print("Starting Policy Iteration")
    print("=" * 50)
    
    optimal_policy, final_v = policy_iteration(env, gamma=0.01, theta=1e-3)
    
    print("\n" + "=" * 50)
    print("Environment Details:")
    print("=" * 50)
    print("Goal position:", env.goal)
    print("Start position:", start_pos)
    print("Barrier positions:")
    barrier_count = 0
    for i in range(env.n):
        for j in range(env.n):
            if env.rewards[i, j] < -1:
                print(f"  Barrier {barrier_count+1} at ({i}, {j})")
                barrier_count += 1
    
    # Save value function
    save_value_function(env, final_v, f"output/value_function2.png")
    
    # Save policy
    save_policy(env, optimal_policy, "output/policy2.png")
    
    # Record simulation video
    simulate_and_record(env, optimal_policy, "output/gridworld_simulation2.mp4")
    
    print("\n📁 All files saved in the 'output' directory:")