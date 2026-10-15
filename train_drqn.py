

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from drqn_model import DRQN


# ──────────────────────────────────────────────────────────────────────
# Goal-balanced episode replay buffer
# Always includes step 0 (the hint step) in every sampled sequence.
# ──────────────────────────────────────────────────────────────────────

class EpisodeReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer_A = deque(maxlen=capacity // 2)
        self.buffer_B = deque(maxlen=capacity // 2)

    def push(self, episode: list, goal_is_A: bool) -> None:
        if goal_is_A:
            self.buffer_A.append(episode)
        else:
            self.buffer_B.append(episode)

    def _from_start(self, ep: list, seq_len: int) -> list:
        """Always start from step 0 so the hint is always present."""
        if len(ep) >= seq_len:
            return list(ep[:seq_len])
        # Pad short episodes at the end
        last_next_obs = ep[-1][3]
        pad = (last_next_obs, 0, 0.0, last_next_obs, 1.0)
        return list(ep) + [pad] * (seq_len - len(ep))

    def sample(self, batch_size: int, seq_len: int) -> list:
        half = batch_size // 2
        avail_A = list(self.buffer_A)
        avail_B = list(self.buffer_B)

        if len(avail_A) < 1 or len(avail_B) < 1:
            combined = avail_A + avail_B
            if not combined:
                return []
            return [self._from_start(random.choice(combined), seq_len)
                    for _ in range(batch_size)]

        seqs = (
            [self._from_start(random.choice(avail_A), seq_len)
             for _ in range(half)]
            + [self._from_start(random.choice(avail_B), seq_len)
               for _ in range(batch_size - half)]
        )
        random.shuffle(seqs)
        return seqs

    def n_goal_episodes(self) -> int:
        return len(self.buffer_A)

    def __len__(self) -> int:
        return len(self.buffer_A) + len(self.buffer_B)


# ──────────────────────────────────────────────────────────────────────
# Warmup: fill buffer with balanced goal episodes
# ──────────────────────────────────────────────────────────────────────

def _warmup_buffer(env, model, buffer, min_goal_eps, max_warmup_eps,
                   action_dim, device, verbose):
    ep = 0
    while buffer.n_goal_episodes() < min_goal_eps and ep < max_warmup_eps:
        obs = env.reset()
        hidden = model.init_hidden(1, device=device)
        ep_mem = []
        goal_is_A = (int(env.active_goal[0]) == int(env.goal_A[0])
                     and int(env.active_goal[1]) == int(env.goal_A[1]))

        while True:
            state = torch.tensor(obs, dtype=torch.float32,
                                 device=device).view(1, 1, -1)
            with torch.no_grad():
                _, hidden = model(state,
                                  (hidden[0].detach(), hidden[1].detach()))

            # 50% goal-directed bias to seed successful episodes faster
            if np.random.rand() < 0.5 and env.active_goal is not None:
                g = env.active_goal
                dr = int(g[0]) - int(env.state[0])
                dc = int(g[1]) - int(env.state[1])
                action = ((1 if dr > 0 else 0) if abs(dr) >= abs(dc)
                          else (3 if dc > 0 else 2))
            else:
                action = random.randint(0, action_dim - 1)

            next_obs, reward, done = env.step(action)
            ep_mem.append((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            if done:
                break

        buffer.push(ep_mem, goal_is_A)
        ep += 1

    if verbose:
        print(f"  [warmup] {ep} episodes  |  "
              f"{len(buffer.buffer_A)} goal_A  |  "
              f"{len(buffer.buffer_B)} goal_B in buffer")


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_drqn(
    env,
    episodes:           int = 3000,
    hidden_dim:         int = 256,
    lr:                 float = 3e-4,
    lr_decay:           float = 0.9998,
    gamma:              float = 0.99,
    burn_in:            int = 5,      # FIX 3: was 10, shorter gradient path
    train_len:          int = 20,     # FIX 3: was 15
    batch_size:         int = 32,     # FIX 6: was 16
    buffer_size:        int = 5_000,
    min_goal_eps:       int = 10,
    max_warmup_eps:     int = 1_000,
    eps_start:          float = 1.0,
    eps_min:            float = 0.02,   # FIX 4: was 0.05
    eps_decay:          float = 0.9985,  # FIX 4: was 0.994, much slower now
    target_update_freq: int = 20,     # FIX 5: was 50
    train_every:        int = 1,
    updates_per_train:  int = 4,
    grad_clip:          float = 5.0,
    aux_weight:         float = 2.0,   # BUG 8 FIX: was 20.0, now balanced
    aux_steps:          int = 5,
    device_str:         str = "cpu",
    seed:               int = 0,
    verbose:            bool = True,
) -> tuple:
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str)
    seq_len = burn_in + train_len
    obs_dim = env.obs_dim
    action_dim = env.action_space

    model = DRQN(obs_dim, hidden_dim, action_dim).to(device)
    target_model = DRQN(obs_dim, hidden_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    q_loss_fn = nn.SmoothL1Loss()
    buffer = EpisodeReplayBuffer(capacity=buffer_size)

    if verbose:
        print("  [warmup] seeding buffer with balanced goal episodes …")
    _warmup_buffer(env, model, buffer, min_goal_eps, max_warmup_eps,
                   action_dim, device, verbose)

    epsilon = eps_start
    rewards_hist = []
    running_avg = deque(maxlen=20)

    for ep in range(episodes):
        obs = env.reset()
        hidden = model.init_hidden(1, device=device)
        ep_mem = []
        total = 0.0
        goal_is_A = (int(env.active_goal[0]) == int(env.goal_A[0])
                     and int(env.active_goal[1]) == int(env.goal_A[1]))

        # ── Rollout ──────────────────────────────────────────────────
        while True:
            hidden = (hidden[0].detach(), hidden[1].detach())
            state = torch.tensor(obs, dtype=torch.float32,
                                 device=device).view(1, 1, -1)
            with torch.no_grad():
                q_vals, hidden = model(state, hidden)

            action = (random.randint(0, action_dim - 1)
                      if random.random() < epsilon
                      else int(q_vals[0, -1].argmax().item()))

            next_obs, reward, done = env.step(action)
            ep_mem.append((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            total += reward
            if done:
                break

        rewards_hist.append(total)
        running_avg.append(total)
        buffer.push(ep_mem, goal_is_A)

        # ── Training ─────────────────────────────────────────────────
        if ep % train_every == 0 and len(buffer) >= batch_size:
            model.train()
            for _ in range(updates_per_train):
                batch = buffer.sample(batch_size, seq_len)
                if not batch:
                    continue

                optimizer.zero_grad()

                # ── FIX 1: torch.tensor accumulators, NOT Python 0.0 ──
                total_q_loss = torch.tensor(0.0, device=device)
                total_aux_loss = torch.tensor(0.0, device=device)

                for seq in batch:
                    obs_arr = np.array([s[0] for s in seq])
                    nobs_arr = np.array([s[3] for s in seq])

                    train_seq = seq[burn_in:]
                    acts = torch.tensor([s[1] for s in train_seq],
                                        dtype=torch.long, device=device)
                    rews = torch.tensor([s[2] for s in train_seq],
                                        dtype=torch.float32, device=device)
                    dones_t = torch.tensor([s[4] for s in train_seq],
                                           dtype=torch.float32, device=device)

                    states_f = torch.tensor(obs_arr,  dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    # nstates_f is kept only for reference; we no longer pass
                    # it to the target network (that was Bug 7).

                    h0 = model.init_hidden(1, device=device)

                    # Online forward — full sequence WITH gradients
                    q_full, _, lstm_out = model(states_f, h0,
                                                return_lstm_out=True)

                    
                    with torch.no_grad():
                        h0t = target_model.init_hidden(1, device=device)
                        q_target_full, _ = target_model(states_f, h0t)
                        # Shift: q_next[t] = q_target_full[t+1]
                        pad = torch.zeros(1, 1, action_dim, device=device)
                        q_next_shifted = torch.cat(
                            [q_target_full[:, 1:, :], pad], dim=1)

                    # Q-loss on training suffix [burn_in:]
                    q_pred = q_full[0, burn_in:].gather(
                        1, acts.unsqueeze(1)).squeeze(1)
                    q_next_max = q_next_shifted[0,
                                                burn_in:].detach().max(1).values
                    q_target = rews + gamma * q_next_max * (1.0 - dones_t)
                    q_loss = q_loss_fn(q_pred, q_target)

                    # Auxiliary hint-prediction loss on first aux_steps
                    hint_val = float(seq[0][0][9])   # obs dim 9 at step 0
                    
                    hint_target = torch.tensor(
                        [1.0 if hint_val > 0 else 0.0], device=device)

                    aux_loss_seq = torch.tensor(0.0, device=device)
                    n_aux = min(aux_steps, lstm_out.size(1))
                    for t in range(n_aux):
                        pred_t = model.hint_head(
                            lstm_out[:, t, :]).squeeze(-1)
                        aux_loss_seq = aux_loss_seq + \
                            F.binary_cross_entropy_with_logits(pred_t,
                                                               hint_target)
                    aux_loss_seq = aux_loss_seq / n_aux

                    
                    total_q_loss = total_q_loss + q_loss
                    total_aux_loss = total_aux_loss + aux_loss_seq

                loss = (total_q_loss / batch_size
                        + aux_weight * total_aux_loss / batch_size)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            model.eval()

        epsilon = max(eps_min, epsilon * eps_decay)
        scheduler.step()

        if ep % target_update_freq == 0:   # FIX 5: every 20 ep
            target_model.load_state_dict(model.state_dict())

        if verbose and ep % 50 == 0:
            avg = float(np.mean(running_avg))
            print(f"[DRQN] Ep {ep:4d}/{episodes}  |  "
                  f"Reward: {total:7.2f}  |  "
                  f"Avg(20): {avg:7.2f}  |  "
                  f"Eps: {epsilon:.3f}  ")

    model.eval()
    return model, rewards_hist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pomdp_gridworld import GridWorldPOMDP

    env = GridWorldPOMDP(wrong_goal_penalty=10.0, step_penalty=0.05)
    model, rewards = train_drqn(env, episodes=3000)

    window = 20
    smoothed = [np.mean(rewards[max(0, i-window+1):i+1])
                for i in range(len(rewards))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,  alpha=0.25, color="darkorange", label="Raw")
    plt.plot(smoothed, color="darkorange", linewidth=2, label=f"Avg-{window}")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5,
                label='DQN ceiling (~0)')
    plt.title("DRQN")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("drqn.png", dpi=150)
    print("Saved: drqn.png")
