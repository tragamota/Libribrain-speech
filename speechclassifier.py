import torch
from torch import nn


class SoftSensorAttention(nn.Module):

    def __init__(self, in_channels=306, time_steps=200, num_heads=4, hidden_dim=256, num_layers=2):
        super(SoftSensorAttention, self).__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=time_steps,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.3,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        weights = self.proj(x)
        weights = weights.unsqueeze(2)

        return x * weights


class SpeechClassifier(nn.Module):
    valid_modes = ['contrastive', 'classification']

    def __init__(self, mode="contrastive"):
        super(SpeechClassifier, self).__init__()

        assert mode in self.valid_modes, f"Invalid mode '{mode}'. Choose from {self.valid_modes}"

        self.mode = mode

        self.sensor_attention = SoftSensorAttention(in_channels=306)

        self.encoder = nn.Sequential(
            nn.Conv1d(306, 250, kernel_size=7, padding=3),
            nn.BatchNorm1d(250),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(250, 184, kernel_size=5, padding=2),
            nn.BatchNorm1d(184),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(184, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.pos_embed = nn.Embedding(200, 128)

        self.attention_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=4, activation='gelu', dim_feedforward=512, dropout=0.3, batch_first=True), num_layers=4, norm=nn.LayerNorm(128))

        self.projection = nn.Sequential(
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Linear(96, 128),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Linear(96, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        _, _, seq_len = x.size()

        x = self.sensor_attention(x)

        x = self.encoder(x)
        x = x.permute(0, 2, 1)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = x + self.pos_embed(positions)
        x = self.attention_encoder(x)

        logits = self.classifier(x)

        if self.mode == "contrastive":
            embedding = self.projection(x)

            return logits, embedding

        return logits


class SpeechClassifierSTFT(nn.Module):
    valid_modes = ['contrastive', 'classification']

    def __init__(self, mode="contrastive"):
        super(SpeechClassifierSTFT, self).__init__()

        assert mode in self.valid_modes, f"Invalid mode '{mode}'. Choose from {self.valid_modes}"

        self.mode = mode

        self.encoder = nn.Sequential(
            nn.Conv2d(306, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Conv2d(64, 200, kernel_size=1),
            nn.BatchNorm2d(200),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.pos_embed = nn.Embedding(200, 64)
        self.attention_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, activation='gelu', dim_feedforward=256, dropout=0.3, batch_first=True),
            num_layers=4,
            norm=nn.LayerNorm(64))

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        # x = x.permute(0, 2, 1)

        positions = torch.arange(200, device=x.device).unsqueeze(0)

        x = x + self.pos_embed(positions)
        x = self.attention_encoder(x)

        return self.classifier(x)
