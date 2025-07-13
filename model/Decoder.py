class Decoder(torch.nn.Module):
  def __init__(self, embed_size, output_size=41):
    super().__init__()

    self.mlp = nn.Sequential(
        PermuteBlock(), nn.BatchNorm1d(2*embed_size), PermuteBlock(),
        torch.nn.Linear(2*embed_size, 256),
        PermuteBlock(), nn.BatchNorm1d(256), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.15),
        torch.nn.Linear(256, 256),
        PermuteBlock(), nn.BatchNorm1d(256), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.15),
        torch.nn.Linear(256, 256),
        PermuteBlock(), nn.BatchNorm1d(256), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.15),
        torch.nn.Linear(256, 256),
        PermuteBlock(), nn.BatchNorm1d(256), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.15),
        torch.nn.Linear(256, 128),
        PermuteBlock(), nn.BatchNorm1d(128), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.15),
        torch.nn.Linear(128, 128),
        PermuteBlock(), nn.BatchNorm1d(128), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
        PermuteBlock(), nn.BatchNorm1d(64), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Linear(64, 128),
        PermuteBlock(), nn.BatchNorm1d(128), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
        PermuteBlock(), nn.BatchNorm1d(64), PermuteBlock(),
        torch.nn.GELU(),
        torch.nn.Linear(64, output_size)
    )

    self.softmax = nn.LogSoftmax(dim=2)

    self._init_weights()

  def forward(self, encoder_out):
    out = self.mlp(encoder_out)
    out = self.softmax(out)

    return out

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
