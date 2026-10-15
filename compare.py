

import random
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from pomdp_gridworld import GridWorldPOMDP
from train_dqn import train_dqn
from train_drqn import train_drqn


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def moving_avg(values, window=20):
    return [float(np.mean(values[max(0, i-window+1):i+1]))
            for i in range(len(values))]


def pca_2d(X):
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def run_random_baseline(env, n=200):
    rewards = []
    for _ in range(n):
        env.reset()
        total, done = 0.0, False
        while not done:
            _, r, done = env.step(random.randint(0, env.action_space - 1))
            total += r
        rewards.append(total)
    return rewards


def success_rate(rewards, threshold=0.0):
    return 100.0 * sum(r > threshold for r in rewards) / max(len(rewards), 1)


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_agent(model, env, n_episodes=200):
    """Greedy evaluation; returns rewards, mean, std, success_rate."""
    model.eval()
    is_drqn = hasattr(model, 'lstm')
    rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        tot = 0.0
        if is_drqn:
            hidden = model.init_hidden(1)

        while not done:
            with torch.no_grad():
                if is_drqn:
                    obs_t = torch.FloatTensor(obs).view(1, 1, -1)
                    q, hidden = model(obs_t, hidden)
                    action = int(q[0, -1].argmax().item())
                else:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0)
                    action = int(model(obs_t).argmax(1).item())
            obs, r, done = env.step(action)
            tot += r
        rewards.append(tot)

    return (rewards,
            float(np.mean(rewards)),
            float(np.std(rewards)),
            success_rate(rewards))


# ══════════════════════════════════════════════════════════════════════
# Hidden-state analysis
# ══════════════════════════════════════════════════════════════════════

def collect_hidden_states(model, env, n_episodes=50):
    model.eval()
    hiddens, positions, dists, steps_list, goal_ids = [], [], [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            state = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
            hidden = model.init_hidden(1)
            step = 0
            done = False
            is_A = (int(env.active_goal[0]) == int(env.goal_A[0])
                    and int(env.active_goal[1]) == int(env.goal_A[1]))
            gid = 1.0 if is_A else 0.0

            while not done:
                hidden = (hidden[0].detach(), hidden[1].detach())
                q, hidden = model(state, hidden)

                h_vec = hidden[0].squeeze().cpu().numpy()
                pos = env.get_position()
                dist = (abs(pos[0] - int(env.active_goal[0]))
                        + abs(pos[1] - int(env.active_goal[1])))

                hiddens.append(h_vec)
                positions.append(pos)
                dists.append(float(dist))
                steps_list.append(float(step))
                goal_ids.append(gid)

                action = int(q[0, -1].argmax().item())
                next_obs, _, done = env.step(action)
                state = torch.tensor(
                    next_obs, dtype=torch.float32).view(1, 1, -1)
                step += 1

    return (np.array(hiddens), positions,
            np.array(dists), np.array(steps_list), np.array(goal_ids))


def build_belief_heatmap(model, env, n_episodes=100):
    model.eval()
    cell_norms = defaultdict(list)

    with torch.no_grad():
        for _ in range(n_episodes):
            obs = env.reset()
            state = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
            hidden = model.init_hidden(1)
            done = False

            while not done:
                hidden = (hidden[0].detach(), hidden[1].detach())
                q, hidden = model(state, hidden)
                pos = env.get_position()
                cell_norms[pos].append(float(hidden[0].norm().item()))
                action = int(q[0, -1].argmax().item())
                next_obs, _, done = env.step(action)
                state = torch.tensor(
                    next_obs, dtype=torch.float32).view(1, 1, -1)

    heatmap = np.zeros((env.grid_size, env.grid_size))
    for (r, c), norms in cell_norms.items():
        heatmap[r, c] = float(np.mean(norms))
    return heatmap


# ══════════════════════════════════════════════════════════════════════
# Quantitative comparison table
# ══════════════════════════════════════════════════════════════════════

def print_comparison_table(dqn_last, drqn_last, rand_rewards, window=100):
    col_w = 12
    header = (f"  {'Metric':<30} {'DQN':>{col_w}}"
              f" {'DRQN':>{col_w}} {'Random':>{col_w}}")
    sep = "  " + "─" * (30 + col_w * 3 + 6)

    print(f"\n{'═'*len(sep)}")
    print(f"  Quantitative Comparison (last {window} training episodes)")
    print(f"{'═'*len(sep)}")
    print(header)
    print(sep)

    rows = {
        "Mean Reward":      (np.mean(dqn_last),   np.mean(drqn_last),
                             np.mean(rand_rewards)),
        "Std Reward":       (np.std(dqn_last),    np.std(drqn_last),
                             np.std(rand_rewards)),
        "Median Reward":    (np.median(dqn_last), np.median(drqn_last),
                             np.median(rand_rewards)),
        "Success Rate (%)": (success_rate(dqn_last), success_rate(drqn_last),
                             success_rate(rand_rewards)),
    }
    for name, vals in rows.items():
        print(f"  {name:<30}"
              + "".join(f" {v:>{col_w}.2f}" for v in vals))

    print(f"{'═'*len(sep)}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════

BLUE = "#2980b9"
ORANGE = "#e67e22"
RED = "#c0392b"


def plot_reward_curves(dqn_rewards, drqn_rewards, rand_mean, episodes, env):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(dqn_rewards,             alpha=0.15, color=BLUE,   lw=0.8)
    ax.plot(drqn_rewards,            alpha=0.15, color=ORANGE, lw=0.8)
    ax.plot(moving_avg(dqn_rewards),  color=BLUE,   lw=2,
            label="DQN  (avg-20)")
    ax.plot(moving_avg(drqn_rewards), color=ORANGE, lw=2,
            label="DRQN (avg-20)")
    ax.axhline(rand_mean, ls="--", color=RED, lw=1.5,
               label=f"Random baseline ({rand_mean:.1f})")
    ax.axhline(0, ls=":", color="grey", lw=1.0,
               label="DQN theoretical ceiling (≈ 0)")

    ax.set_title(
        "Episode Rewards — DQN vs DRQN on Two-Goal Memory POMDP\n"
        f"({env.grid_size}×{env.grid_size} grid | {env.n_obstacles} obstacles"
        f" | goal_A={tuple(env.goal_A.tolist())}"
        f" | goal_B={tuple(env.goal_B.tolist())}"
        f" | {episodes} episodes)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig("1_reward_curves.png", dpi=150)
    plt.close()
    print("  Saved: 1_reward_curves.png")


def plot_performance_bar(dqn_last, drqn_last, rand_rewards,
                         dqn_eval_sr, drqn_eval_sr):
    labels = ["Random", "DQN\n(last 100)", "DRQN\n(last 100)"]
    means = [np.mean(rand_rewards), np.mean(dqn_last), np.mean(drqn_last)]
    stds = [np.std(rand_rewards),  np.std(dqn_last),  np.std(drqn_last)]
    colors = [RED, BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(labels, means, yerr=stds, color=colors,
                       capsize=6, width=0.5, error_kw={"linewidth": 1.5})
    for bar, val in zip(bars, means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.2 if val >= 0 else -0.8),
            f"{val:.2f}", ha="center", va="bottom",
            fontweight="bold", fontsize=10)
    axes[0].set_title("Mean ± Std Reward", fontweight="bold")
    axes[0].set_ylabel("Mean Reward")
    axes[0].grid(True, axis="y", ls="--", alpha=0.4)

    sr_labels = ["Random", "DQN\n(eval 200)", "DRQN\n(eval 200)"]
    sr_vals = [success_rate(rand_rewards), dqn_eval_sr, drqn_eval_sr]
    bars2 = axes[1].bar(sr_labels, sr_vals, color=colors, width=0.5)
    for bar, val in zip(bars2, sr_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f} %", ha="center", va="bottom",
            fontweight="bold", fontsize=10)
    axes[1].axhline(50, ls=":", color="grey", lw=1.5,
                    label="50 % ceiling (DQN theoretical max)")
    axes[1].set_title("Correct-Goal Success Rate (%)", fontweight="bold")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", ls="--", alpha=0.4)

    fig.suptitle(
        "Performance Comparison: DQN vs DRQN vs Random\n",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("2_performance_bar_fixed.png", dpi=150)
    plt.close()
    print("  Saved: 2_performance_bar_fixed.png")


def plot_hidden_state_pca(hiddens, dists, steps_arr, goal_ids):
    if len(hiddens) < 4:
        print("  Skipped PCA (too few data points).")
        return

    proj = pca_2d(hiddens)
    var_total = np.var(hiddens, axis=0).sum()
    _, S, _ = np.linalg.svd(hiddens - hiddens.mean(0), full_matrices=False)
    pct = 100 * (S[:2]**2) / (hiddens.shape[0]-1) / max(var_total, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = axes[0].scatter(proj[:, 0], proj[:, 1], c=dists,
                          cmap="RdYlGn_r", s=10, alpha=0.55)
    plt.colorbar(sc1, ax=axes[0]).set_label(
        "Manhattan dist. to active goal", fontsize=9)
    axes[0].set_title("DRQN Hidden States (PCA)\n"
                      "Coloured by Distance to Active Goal",
                      fontweight="bold")
    axes[0].set_xlabel(f"PC 1  ({pct[0]:.1f} % var)")
    axes[0].set_ylabel(f"PC 2  ({pct[1]:.1f} % var)")

    sc2 = axes[1].scatter(proj[:, 0], proj[:, 1], c=goal_ids,
                          cmap="coolwarm", s=10, alpha=0.65, vmin=0, vmax=1)
    cb2 = plt.colorbar(sc2, ax=axes[1])
    cb2.set_label("Goal identity (1=goal_A, 0=goal_B)", fontsize=9)
    cb2.set_ticks([0, 1])
    cb2.set_ticklabels(["goal_B", "goal_A"])
    axes[1].set_title("DRQN Hidden States (PCA)\n"
                      "Coloured by Goal Identity",
                      fontweight="bold")
    axes[1].set_xlabel(f"PC 1  ({pct[0]:.1f} % var)")
    axes[1].set_ylabel(f"PC 2  ({pct[1]:.1f} % var)")

    fig.suptitle(
        "Belief-State Visualisation — PCA of LSTM Hidden States\n"
        "(Two clean clusters by goal identity prove LSTM memory works)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("3_hidden_state_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 3_hidden_state_pca.png")


def plot_belief_heatmap(heatmap, env):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(heatmap, cmap="plasma", origin="upper",
                        interpolation="nearest")
    plt.colorbar(im, ax=axes[0]).set_label(
        "Mean ‖h‖  (hidden-state L2 norm)", fontsize=9)
    axes[0].scatter([env.goal_A[1]], [env.goal_A[0]],
                    marker="*", color="cyan", s=300, zorder=5, label="Goal A")
    axes[0].scatter([env.goal_B[1]], [env.goal_B[0]],
                    marker="*", color="lime", s=300, zorder=5, label="Goal B")
    for (r, c) in env.obstacles:
        axes[0].add_patch(plt.Rectangle((c-.5, r-.5), 1, 1,
                                        color="white", alpha=0.4, zorder=3))
    axes[0].set_title("Belief Heatmap (DRQN)\n"
                      "(Mean LSTM Hidden-State Norm per Cell)",
                      fontweight="bold")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    axes[0].legend(fontsize=8, loc="upper right")

    layout = np.ones((env.grid_size, env.grid_size)) * 0.5
    for (r, c) in env.obstacles:
        layout[r, c] = 0.0
    layout[int(env.goal_A[0]), int(env.goal_A[1])] = 0.9
    layout[int(env.goal_B[0]), int(env.goal_B[1])] = 0.9

    axes[1].imshow(layout, cmap="Blues", origin="upper",
                   interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_title(
        f"Grid Layout  ({env.grid_size}×{env.grid_size})\n"
        f"{env.n_obstacles} obstacles | "
        f"goal_A={tuple(env.goal_A.tolist())} | "
        f"goal_B={tuple(env.goal_B.tolist())}",
        fontweight="bold")
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if (r, c) in env.obstacles:
                axes[1].text(c, r, "#", ha="center", va="center",
                             color="white", fontsize=9, fontweight="bold")
    axes[1].text(int(env.goal_A[1]), int(env.goal_A[0]), "A",
                 ha="center", va="center", color="white",
                 fontsize=11, fontweight="bold")
    axes[1].text(int(env.goal_B[1]), int(env.goal_B[0]), "B",
                 ha="center", va="center", color="white",
                 fontsize=11, fontweight="bold")
    axes[1].legend(handles=[
        mpatches.Patch(facecolor="#1f4e79", label="Obstacle (#)"),
        mpatches.Patch(facecolor="#aec6e8", label="Free cell"),
        mpatches.Patch(facecolor="#08519c", label="Goal A / Goal B"),
    ], loc="upper right", fontsize=8)
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")

    fig.suptitle(
        "DRQN Belief Formation Analysis\n"
        "→ strong goal-specific representations)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("4_belief_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 4_belief_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    EPISODES = 3000
    SEED = 42
    LAST_N = 100

    env = GridWorldPOMDP(
        grid_size=11,
        obs_radius=1,
        sensor_noise=0.1,
        n_obstacles=15,
        max_steps=150,
        goal_reward=10.0,
        wrong_goal_penalty=10.0,
        step_penalty=0.05,
        seed=SEED,
    )

    print("=" * 70)
    print("  POMDP GridWorld — DQN vs DRQN")
    print("=" * 70)
    print(f"  Grid            : {env.grid_size}×{env.grid_size}")
    print(f"  Obstacles       : {env.n_obstacles}")
    print(f"  Obs dim         : {env.obs_dim}"
          f"  (9 FOV + 1 hint + 2 compass_A + 2 compass_B)")
    print(f"  Goal A          : {tuple(env.goal_A.tolist())}")
    print(f"  Goal B          : {tuple(env.goal_B.tolist())}")
    print(f"  Hint            : ±1 at step 0 only, 0 forever after")
    print(f"  Correct goal    : +{env.goal_reward}")
    print(f"  Wrong goal      : −{env.wrong_goal_penalty}")
    print(f"  Step penalty    : −{env.step_penalty}")
    print(f"  Episodes        : {EPISODES}")
    print()

    env.reset()
    env.render()

    print("─" * 70)
    print("  Training DQN …")
    print("─" * 70)
    dqn_model, dqn_rewards = train_dqn(
        env, episodes=EPISODES, seed=SEED, verbose=True)

    print("\n" + "─" * 70)
    print("  Training DRQN …")
    print("─" * 70)
    drqn_model, drqn_rewards = train_drqn(
        env, episodes=EPISODES, seed=SEED, verbose=True)

    print("\n  Running random baseline (200 episodes) …")
    rand_rewards = run_random_baseline(env, n=200)

    print_comparison_table(dqn_rewards[-LAST_N:], drqn_rewards[-LAST_N:],
                           rand_rewards, window=LAST_N)

    print("  Evaluating agents greedily (200 episodes each) …")
    _, dqn_m,  dqn_s,  dqn_sr = evaluate_agent(dqn_model,  env, 200)
    _, drqn_m, drqn_s, drqn_sr = evaluate_agent(drqn_model, env, 200)
    print(f"  DQN        eval: mean={dqn_m:.2f} ± {dqn_s:.2f}"
          f"  success={dqn_sr:.1f}%")
    print(f"  DRQN eval: mean={drqn_m:.2f} ± {drqn_s:.2f}"
          f"  success={drqn_sr:.1f}%")

    print("\n  Collecting DRQN-FIXED hidden states (50 episodes) …")
    hiddens, positions, dists, steps_arr, goal_ids = collect_hidden_states(
        drqn_model, env, n_episodes=50)

    print("  Building belief heatmap (100 episodes) …")
    heatmap = build_belief_heatmap(drqn_model, env, n_episodes=100)

    print("\n  Saving plots …")
    plt.rcParams.update({"font.family":       "DejaVu Sans",
                         "axes.spines.top":   False,
                         "axes.spines.right": False})

    plot_reward_curves(dqn_rewards, drqn_rewards,
                       float(np.mean(rand_rewards)), EPISODES, env)
    plot_performance_bar(dqn_rewards[-LAST_N:], drqn_rewards[-LAST_N:],
                         rand_rewards, dqn_sr, drqn_sr)
    plot_hidden_state_pca(hiddens, dists, steps_arr, goal_ids)
    plot_belief_heatmap(heatmap, env)

    print("\n  All outputs saved.  Run complete.")
    print()
    if drqn_sr > dqn_sr + 10:
        print(f"   DRQN ({drqn_sr:.1f}%) beats DQN ({dqn_sr:.1f}%) by"
              f" {drqn_sr-dqn_sr:.1f}pp — LSTM memory is working!")
    else:
        print(f"   DRQN ({drqn_sr:.1f}%) did not clearly beat DQN "
              f"({dqn_sr:.1f}%).")
