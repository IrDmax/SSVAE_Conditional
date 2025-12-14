import torch.nn as nn

class ImmuneLSTM(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.D = 2 if bidirectional else 1
        
        # Predict next-step latent embedding
        self.predict_next_z = nn.Linear(self.D * hidden_dim, latent_dim)
        
        # Predict clinical outcome
        self.outcome_head = nn.Sequential(
            nn.Linear(self.D * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # binary outcome
            nn.Sigmoid()
        )

    def forward(self, z_sequence):
        """
        z_sequence: (batch, time_steps=4, latent_dim)
        """
        lstm_out, (h_last, c_last) = self.lstm(z_sequence)
        

        h_final = lstm_out[:, -1, :]  # (batch, D*hidden_dim)

        # Predict the next immune state embedding
        next_z = self.predict_next_z(h_final)

        # Predict clinical outcome (mild, severe, critical)
        outcome = self.outcome_head(h_final)

        return {
            "h_final": h_final,
            "next_z": next_z,
            "outcome": outcome
        }

