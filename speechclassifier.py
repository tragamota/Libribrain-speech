from torch import nn

class SoftSensorAttention(nn.Module):

    def __init__(self):
        super(SoftSensorAttention, self).__init__()

    def forward(self, x):

        return x


class SpeechClassifier(nn.Module):
    valid_modes = ['contrastive', 'classification']

    def __init__(self, mode="contrastive"):
        super(SpeechClassifier, self).__init__()

        assert mode in self.valid_modes, f"Invalid mode '{mode}'. Choose from {self.valid_modes}"

        self.mode = mode

        self.encoder = nn.Sequential(
            nn.Conv1d(306, 512, kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, 384, kernel_size=5, padding=2),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.attention_downsample = nn.Sequential(
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, activation='gelu', nhead=8, dim_feedforward=256, dropout=0.4, batch_first=True), num_layers=2),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, activation='gelu', nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True), num_layers=2)
        )

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        embedding = self.attention_downsample(x)
        x = self.classifier(embedding)

        if self.mode == "contrastive":
            embedding = self.projection(embedding)

            return x, embedding

        return x

