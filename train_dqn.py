

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn_model import DQN
from repbuf_dqn import ReplayBuffer


def train_dqn(
    env,
    episodes:           int = 3000,
    hidden_dim:         int = 256,
    lr:                 float = 1e-3,
    gamma:              float = 0.99,
    batch_size:         int = 64,
    buffer_size:        int = 10_000,
    eps_start:          float = 1.0,
    eps_min:            float = 0.05,
    eps_decay:          float = 0.995,
    target_update_freq: int = 10,
    seed:               int = 0,
    verbose:            bool = True,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs_dim = env.obs_dim
    action_dim = env.action_space

    model = DQN(obs_dim, hidden_dim, action_dim)
    target_model = DQN(obs_dim, hidden_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size)
    epsilon = eps_start
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, action_dim)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0)
                    action = model(st).argmax(1).item()

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(
                    batch_size)
                st = torch.FloatTensor(states)
                act = torch.LongTensor(actions)
                rew = torch.FloatTensor(rewards)
                nst = torch.FloatTensor(next_states)
                don = torch.FloatTensor(dones)

                q_pred = model(st).gather(1, act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_model(nst).max(1)[0]
                    q_target = rew + gamma * q_next * (1.0 - don)

                loss = F.smooth_l1_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        rewards_history.append(total_reward)

        if ep % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(eps_min, epsilon * eps_decay)

        if verbose and ep % 50 == 0:
            avg = (np.mean(rewards_history[-20:]) if len(rewards_history) >= 20
                   else np.mean(rewards_history))
            print(f"[DQN]  Ep {ep:4d}/{episodes}  |  "
                  f"Reward: {total_reward:7.2f}  |  "
                  f"Avg(20): {avg:7.2f}  |  Eps: {epsilon:.3f}")

    model.eval()
    return model, rewards_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pomdp_gridworld import GridWorldPOMDP

    env = GridWorldPOMDP(wrong_goal_penalty=10.0, step_penalty=0.05)
    model, rewards = train_dqn(env, episodes=3000)
    window = 20
    smoothed = [np.mean(rewards[max(0, i-window+1):i+1])
                for i in range(len(rewards))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,  alpha=0.25, color="royalblue", label="Raw")
    plt.plot(smoothed, color="royalblue", linewidth=2, label=f"Avg-{window}")
    plt.title("DQN — Two-Goal Memory POMDP v9")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("dqn_standalone.png", dpi=150)
    print("Saved: dqn_standalone.png")
