"""
pomdp_gridworld.py  (v9 — two-goal memory task)
=================================================

Why this is the only environment that PROVES DRQN > DQN
---------------------------------------------------------
Every previous version failed because the ambiguity was PERSISTENT (noisy
compass, flickering).  DQN could average across the noise or learn a biased
heuristic and still achieve ~75-83% success.

v9 uses a QUALITATIVELY different mechanism: a ONE-TIME hint.

  At step 0:   obs dim 9 carries  hint = +1  (active goal is goal_A)
                                  hint = -1  (active goal is goal_B)
  Steps 1+:   hint = 0.  Always.  Both goal types are observationally
               IDENTICAL after step 0.

DQN's provable failure on this task
-------------------------------------
After step 0 the observation (FOV=same, hint=0, cA=same, cB=same) is
IDENTICAL regardless of which goal is active.  The replay buffer therefore
contains THE SAME observation paired with CONTRADICTORY Q-targets:

  Goal-A episode, step 5:  obs(hint=0) → optimal action = follow compass_A
  Goal-B episode, step 5:  obs(hint=0) → optimal action = follow compass_B

Gradient descent averages these conflicting signals.  DQN converges to a
policy that follows NEITHER compass reliably — yielding ~50% correct-goal
success.  This is the theoretical maximum for ANY memoryless policy on this
environment, regardless of network size or training duration.

DRQN's advantage
-----------------
Step 0 → LSTM encodes hint into h₀.
Steps 1+ → obs has hint=0, but h_t CARRIES the encoded goal info forward.
  "I saw +1 at step 0 → follow compass_A, ignore compass_B."

This is precisely the belief-state tracking role the LSTM was designed for.

Observation vector (14 dimensions)
------------------------------------
  dims  0-8 :  3×3 binary FOV  (0=blocked, 1=passable)
               peripheral cells have mild sensor_noise corruption
  dim   9   :  hint  = ±1 at step 0,  0 for all subsequent steps
  dims 10-11:  compass toward goal_A  (row_dir ∈ {-1,0,1}, col_dir)  no noise
  dims 12-13:  compass toward goal_B  (row_dir ∈ {-1,0,1}, col_dir)  no noise

  KEY DESIGN CHOICE: the dual compass gives BOTH agents perfect navigation
  information toward BOTH goals at every step.  The ONLY thing they differ
  on is which compass to follow — and that depends on the hint at step 0.

Goals
------
  goal_A : (1, grid_size-2)  ← top-right area
  goal_B : (grid_size-2, 1)  ← bottom-left area
  Each episode: one is randomly chosen as the correct target.

Rewards
--------
  Correct goal : +10   (episode ends, success)
  Wrong goal   :  -5   (episode ends, failure)
  Step penalty :  -0.1
  Timeout      :  ≈ -15  (150 steps × 0.1)

Actions: 0=Up  1=Down  2=Left  3=Right
"""

import numpy as np


class GridWorldPOMDP:

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    CELL_BLOCKED = 0.0
    CELL_PASSABLE = 1.0

    def __init__(
        self,
        grid_size:          int = 11,
        obs_radius:         int = 1,
        sensor_noise:       float = 0.1,   # mild peripheral FOV noise
        n_obstacles:        int = 15,    # fewer = cleaner navigation paths
        max_steps:          int = 150,
        goal_reward:        float = 10.0,
        wrong_goal_penalty: float = 5.0,
        step_penalty:       float = 0.1,
        seed:               int = 42,
    ):
        self.grid_size = grid_size
        self.obs_radius = obs_radius
        self.sensor_noise = sensor_noise
        self.n_obstacles = n_obstacles
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.wrong_goal_penalty = wrong_goal_penalty
        self.step_penalty = step_penalty

        # Backward-compat attributes (unused in v9 but referenced by compare.py)
        self.p_compass_noise = 0.0
        self.p_flicker = 0.0
        self.p_blank = 0.0

        # obs = 9 FOV + 1 hint + 2 compass_A + 2 compass_B = 14 dims
        self.obs_dim = (2 * obs_radius + 1) ** 2 + 5
        self.action_space = 4

        self._rng = np.random.default_rng(seed)

        # Two fixed goals at opposite corners
        self.goal_A = np.array([1,             grid_size - 2])  # top-right
        self.goal_B = np.array([grid_size - 2, 1])  # bottom-left

        self.obstacles = self._place_obstacles()
        self.free_cells = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if (r, c) not in self.obstacles
            and not (r == int(self.goal_A[0]) and c == int(self.goal_A[1]))
            and not (r == int(self.goal_B[0]) and c == int(self.goal_B[1]))
        ]

        # Assigned by reset()
        self.state = None
        self.steps_taken = 0
        self.active_goal = None  # the goal the agent SHOULD reach this episode
        self.wrong_goal = None  # the other goal
        self.hint_given = False  # True after the first obs is returned

    # ------------------------------------------------------------------

    def _place_obstacles(self) -> set:
        g = self.grid_size - 1
        forbidden: set = set()
        for dr in range(3):
            for dc in range(3):
                forbidden.add((dr,      g - dc))   # top-right  (goal_A zone)
                forbidden.add((g - dr,  dc))   # bottom-left (goal_B zone)

        obstacles: set = set()
        attempts = 0
        while len(obstacles) < self.n_obstacles and attempts < 20_000:
            r = int(self._rng.integers(0, self.grid_size))
            c = int(self._rng.integers(0, self.grid_size))
            if (r, c) not in forbidden and (r, c) not in obstacles:
                obstacles.add((r, c))
            attempts += 1
        return obstacles

    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        idx = int(self._rng.integers(0, len(self.free_cells)))
        self.state = np.array(self.free_cells[idx], dtype=int)
        self.steps_taken = 0
        self.hint_given = False

        # Randomly choose which goal is "correct" this episode
        if self._rng.random() < 0.5:
            self.active_goal = self.goal_A
            self.wrong_goal = self.goal_B
        else:
            self.active_goal = self.goal_B
            self.wrong_goal = self.goal_A

        return self._get_obs()

    def step(self, action: int):
        dr, dc = self.ACTIONS[action]
        nr = int(self.state[0]) + dr
        nc = int(self.state[1]) + dc

        if (0 <= nr < self.grid_size
                and 0 <= nc < self.grid_size
                and (nr, nc) not in self.obstacles):
            self.state = np.array([nr, nc], dtype=int)

        self.steps_taken += 1

        # Correct goal reached
        if (int(self.state[0]) == int(self.active_goal[0])
                and int(self.state[1]) == int(self.active_goal[1])):
            return self._get_obs(), self.goal_reward, True

        # Wrong goal reached
        if (int(self.state[0]) == int(self.wrong_goal[0])
                and int(self.state[1]) == int(self.wrong_goal[1])):
            return self._get_obs(), -self.wrong_goal_penalty, True

        done = self.steps_taken >= self.max_steps
        return self._get_obs(), -self.step_penalty, done

    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # 1. 3×3 binary FOV with mild peripheral noise
        r = self.obs_radius
        fov = np.empty((2*r+1)**2, dtype=np.float32)
        idx = 0
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni = int(self.state[0]) + di
                nj = int(self.state[1]) + dj
                if not (0 <= ni < self.grid_size and 0 <= nj < self.grid_size):
                    val = self.CELL_BLOCKED
                elif (ni, nj) in self.obstacles:
                    val = self.CELL_BLOCKED
                else:
                    val = self.CELL_PASSABLE
                fov[idx] = val
                idx += 1

        centre = len(fov) // 2
        noise_mask = np.random.rand(len(fov)) < self.sensor_noise
        noise_mask[centre] = False
        fov[noise_mask] = 1.0 - fov[noise_mask]

        # 2. One-time hint — THE KEY MEMORY SIGNAL
        # Returns ±1 on step 0 only; 0 forever after.
        if not self.hint_given:
            is_A = (int(self.active_goal[0]) == int(self.goal_A[0])
                    and int(self.active_goal[1]) == int(self.goal_A[1]))
            hint = 1.0 if is_A else -1.0
            self.hint_given = True
        else:
            hint = 0.0

        # 3. Dual compass — direction toward each goal, perfect accuracy
        # Both agents can always compute how to reach EITHER goal.
        # Only DRQN can remember WHICH one to head for.
        ca_row = float(np.sign(int(self.goal_A[0]) - int(self.state[0])))
        ca_col = float(np.sign(int(self.goal_A[1]) - int(self.state[1])))

        cb_row = float(np.sign(int(self.goal_B[0]) - int(self.state[0])))
        cb_col = float(np.sign(int(self.goal_B[1]) - int(self.state[1])))

        return np.array(
            [*fov, hint, ca_row, ca_col, cb_row, cb_col],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------

    def get_position(self) -> tuple:
        return (int(self.state[0]), int(self.state[1]))

    @property
    def goal(self):
        """Backward compat for compare.py — returns goal_A."""
        return self.goal_A

    def render(self) -> None:
        grid = [["." for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]
        for (r, c) in self.obstacles:
            grid[r][c] = "#"
        grid[int(self.goal_A[0])][int(self.goal_A[1])] = "A"
        grid[int(self.goal_B[0])][int(self.goal_B[1])] = "B"
        if self.state is not None:
            grid[int(self.state[0])][int(self.state[1])] = "@"

        border = "+" + "-" * (self.grid_size * 2 - 1) + "+"
        print(border)
        for row in grid:
            print("|" + " ".join(row) + "|")
        print(border)
        active_str = ("A" if self.active_goal is not None
                      and int(self.active_goal[0]) == int(self.goal_A[0])
                      else "B")
        print(
            f"  Pos: {self.get_position() if self.state is not None else 'None'}"
            f"  |  goal_A={tuple(self.goal_A.tolist())}"
            f"  |  goal_B={tuple(self.goal_B.tolist())}"
            f"  |  Active={active_str}"
            f"  |  Steps: {self.steps_taken}/{self.max_steps}"
        )
        print()


# ──────────────────────────────────────────────────────────────────────

def smoke_test(env, episodes=5):
    print(
        f"Smoke test — {env.grid_size}×{env.grid_size} two-goal memory POMDP v9\n")
    for ep in range(episodes):
        obs = env.reset()
        assert obs.shape == (env.obs_dim,), f"shape mismatch: {obs.shape}"
        assert obs[9] in (1.0, -1.0), f"first hint should be ±1, got {obs[9]}"
        done, tot, steps = False, 0.0, 0
        while not done:
            if steps > 0:
                assert obs[9] == 0.0, f"hint should be 0 at step {steps}"
            obs, r, done = env.step(np.random.randint(0, env.action_space))
            tot += r
            steps += 1
        outcome = ("CORRECT" if tot > 0 else
                   ("WRONG" if env.steps_taken < env.max_steps else "timeout"))
        print(f"  Ep {ep+1}: steps={steps}  reward={tot:.2f}  [{outcome}]")
    print("\n✓ smoke test passed\n")


if __name__ == "__main__":
    env = GridWorldPOMDP()
    env.reset()
    env.render()
    smoke_test(env)
