import numpy as np
import random
import gym
from gym import spaces
import pygame
import imageio


# =====================
# GridWorld Gym Environment
# =====================
class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n=5, p_barrier=0.2, r_barrier=-10, seed_nr=0):
        super(GridWorldEnv, self).__init__()
        random.seed(seed_nr)
        np.random.seed(seed_nr)

        self.n = n
        self.action_space = spaces.Discrete(4)  # left, up, right, down
        self.observation_space = spaces.Box(low=0, high=n-1, shape=(2,), dtype=int)

        self.actions = [0, 1, 2, 3]
        self.rewards = (-1) * np.ones((n, n))
        self.goal = (random.randrange(n), random.randrange(n))
        self.agent_pos = (0, 0)

        for i in range(n):
            for j in range(n):
                if (i, j) != self.goal and (i,j) != self.agent_pos :
                    if random.uniform(0, 1) < p_barrier:
                        self.rewards[i, j] = r_barrier


        self.cell_size = 100
        self.screen = None
        self.clock = None

    def reset(self):
        self.agent_pos = (0, 0)
        return np.array(self.agent_pos, dtype=int)

    def step(self, action):
        x, y = self.agent_pos
        n = self.n
        if action == 0:  # left
            y = max(0, y - 1)
        elif action == 1:  # up
            x = max(0, x - 1)
        elif action == 2:  # right
            y = min(n - 1, y + 1)
        elif action == 3:  # down
            x = min(n - 1, x + 1)

        self.agent_pos = (x, y)
        reward = self.rewards[x, y]

        done = (x, y) == self.goal
        if done:
            reward = 10

        return np.array(self.agent_pos, dtype=int), reward, done, {}

    def render(self, mode='human', save_frames=False, frame_list=None):
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
    while not policy_stable:
        policy_evaluation(env, v, pi, gamma, theta)
        policy_stable = policy_improvement(env, v, pi, gamma)
    return pi


def policy_evaluation(env, v, pi, gamma, theta):
    n = env.n
    delta = float("inf")
    while delta > theta:
        delta = 0
        v_old = v.copy()
        for x in range(n):
            for y in range(n):
                if (x, y) == env.goal:
                    continue
                total = 0
                for a in env.actions:
                    s_prime_x, s_prime_y = get_next_state(x, y, a, n)
                    total += pi[x, y, a] * (env.rewards[s_prime_x, s_prime_y] + gamma * v_old[s_prime_x, s_prime_y])
                delta = max(delta, abs(v[x, y] - total))
                v[x, y] = total


def policy_improvement(env, v, pi, gamma):
    policy_stable = True
    n = env.n
    for x in range(n):
        for y in range(n):
            old_pi = pi[x, y, :].copy()
            q_values = np.zeros(4)
            for a in env.actions:
                s_prime_x, s_prime_y = get_next_state(x, y, a, n)
                q_values[a] = env.rewards[s_prime_x, s_prime_y] + gamma * v[s_prime_x, s_prime_y]

            best_actions = np.argwhere(q_values == np.max(q_values)).flatten()
            pi[x, y, :] = 0
            for a in best_actions:
                pi[x, y, a] = 1 / len(best_actions)

            if not np.array_equal(old_pi, pi[x, y, :]):
                policy_stable = False
    return policy_stable


def get_next_state(x, y, a, n):
    if a == 0:
        return x, max(0, y - 1)
    elif a == 1:
        return max(0, x - 1), y
    elif a == 2:
        return x, min(n - 1, y + 1)
    else:
        return min(n - 1, x + 1), y


# =====================
# Run Simulation and Record Video
# =====================
def simulate_and_record(env, pi, filename="gridworld_policy_iteration.mp4"):
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
    print(f"\n🎥 Simulation video saved as {filename}")


# =====================
# Main
# =====================
if __name__ == "__main__":
    env = GridWorldEnv(n=5, p_barrier=0.5, r_barrier=-10, seed_nr=20)
    optimal_policy = policy_iteration(env, gamma=0.9, theta=1e-3)
    simulate_and_record(env, optimal_policy)
