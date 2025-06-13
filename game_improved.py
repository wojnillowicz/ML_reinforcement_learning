import numpy as np
import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import defaultdict

# Environment settings
WIDTH = 7
HEIGHT = 6
ACTIONS = [-1, 0, 1]  # Left, Stay, Right

# Q-learning parameters
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.995


class CatchEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.object_x = random.randint(0, self.width - 1)
        self.basket_x = self.width // 2
        self.step_count = 0
        return self._get_state()

    def step(self, action_idx):
        move = ACTIONS[action_idx]
        self.basket_x = max(0, min(self.width - 1, self.basket_x + move))
        self.step_count += 1

        reward = 0
        done = False
        if self.step_count == self.height:
            done = True
            reward = 1 if self.basket_x == self.object_x else -1

        return self._get_state(), reward, done

    def _get_state(self):
        return (self.object_x, self.basket_x, self.step_count)


class QLearningAgent:
    def __init__(self, actions, alpha, gamma, epsilon):
        self.q_table = defaultdict(lambda: [0.0 for _ in actions])
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        max_next_q = max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[state][action]
        )


def render_game(object_x, basket_x, current_step, delay=0.3, message=None, status=None):
    plt.clf()
    ax = plt.gca()
    ax.set_xlim(-0.5, WIDTH - 0.5)
    ax.set_ylim(-0.5, HEIGHT - 0.5)
    ax.set_xticks(range(WIDTH))
    ax.set_yticks(range(HEIGHT))
    ax.grid(True)

    obj = Circle((object_x, HEIGHT - 1 - current_step), 0.3, color='red')
    ax.add_patch(obj)

    basket = Rectangle((basket_x - 0.4, -0.5), 0.8, 0.8, color='blue')
    ax.add_patch(basket)

    if message:
        plt.text(0.5, 1.05, message, transform=ax.transAxes, ha='center', fontsize=12, color='green')
    if status:
        plt.text(0.5, 1.02, status, transform=ax.transAxes, ha='center', fontsize=10, color='purple')

    plt.pause(delay)


def test_agent(agent, env, delay=0.3, games=5, episode_checkpoint=None):
    plt.ion()
    fig = plt.figure(figsize=(6, 6))

    for _ in range(games):
        state = env.reset()
        done = False
        status = ""

        while not done:
            object_x, basket_x, current_step = state
            render_game(object_x, basket_x, current_step, delay=delay,
                        message=f"Episode: {episode_checkpoint}" if episode_checkpoint else None,
                        status=status)

            action = int(np.argmax(agent.q_table[state]))
            state, reward, done = env.step(action)

            if done:
                status = "🎯 Caught!" if reward == 1 else "❌ Missed!"
                render_game(*state, delay=delay,
                            message=f"Episode: {episode_checkpoint}" if episode_checkpoint else None,
                            status=status)
                time.sleep(1)
                break

    plt.ioff()
    plt.close()


def plot_results(results, total_episodes):
    plt.figure(figsize=(10, 5))
    x_vals = [i * 100 for i in range(1, len(results) + 1)]
    plt.plot(x_vals, results, marker='o')
    plt.title(f"Learning Progress of Q-Learning Agent")
    plt.xlabel("Episodes")
    plt.ylabel("Successful Catches (per 100 episodes)")
    plt.text(0.95, 0.95, f"Total Episodes: {total_episodes}", ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', ec='black', lw=1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_agent(episodes, play_interval, test_games, agent=None, env=None):
    if env is None:
        env = CatchEnvironment(WIDTH, HEIGHT)
    if agent is None:
        agent = QLearningAgent(ACTIONS, ALPHA, GAMMA, EPSILON)

    results = []
    success_counter = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

        if reward == 1:
            success_counter += 1

        if episode % 100 == 0:
            results.append(success_counter)
            success_counter = 0

        if episode % play_interval == 0:
            test_agent(agent, env, games=test_games, episode_checkpoint=episode)

        agent.epsilon = max(MIN_EPSILON, agent.epsilon * DECAY_RATE)

    return agent, env, results


def main_menu():
    agent = None
    env = None
    results = []

    while True:
        print("\n🎮 Q-Learning Catch Game (Matplotlib Version)")
        print("1. Train with live checkpoints")
        print("2. Restart simulation")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            try:
                episodes = int(input("Enter total number of training episodes (e.g. 2000): "))
                play_interval = int(input("Enter interval between test runs (e.g. 500): "))
                test_games = int(input("Enter number of games to display at each test (e.g. 3): "))
                agent, env, results = train_agent(episodes, play_interval, test_games, agent, env)
                print("\n🏁 Final test after training:")
                test_agent(agent, env, games=test_games, episode_checkpoint=episodes)
                plot_results(results, total_episodes=episodes)
            except ValueError:
                print("Invalid input. Please enter integer values.")

        elif choice == "2":
            print("\n🔄 Restarting simulation from scratch...")
            agent = None
            env = None
            results = []

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
