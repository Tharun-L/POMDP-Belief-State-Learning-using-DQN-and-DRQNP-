

import torch
import torch.nn as nn


class DRQN(nn.Module):
    

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)

        # Auxiliary head: predicts which goal is active from the LSTM
        # output at step 0.  Output is a single logit (BCEWithLogitsLoss).
        # 1.0 = goal_A active,  0.0 = goal_B active.
        self.hint_head = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        return_lstm_out: bool = False,
    ) -> tuple:
       
        x = torch.relu(self.encoder(x))    # (batch, seq_len, hidden_dim)
        out, hidden = self.lstm(x, hidden)    # (batch, seq_len, hidden_dim)
        q = self.head(out)                  # (batch, seq_len, action_dim)

        if return_lstm_out:
            return q, hidden, out
        return q, hidden

    # ------------------------------------------------------------------

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)
