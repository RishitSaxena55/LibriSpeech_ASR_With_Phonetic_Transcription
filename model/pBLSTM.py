class pBLSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super(pBLSTM, self).__init__()

    self.blstm1 = nn.LSTM(input_size*2, hidden_size, batch_first=True, bidirectional=True, dropout=0.2)

  def forward(self, x_packed):
    x_unpacked, lens_unpacked = pad_packed_sequence(x_packed, batch_first=True)

    x_reshaped, x_lens_reshaped = self.trunc_reshape(x_unpacked, lens_unpacked)

    x_packed = pack_padded_sequence(x_reshaped, x_lens_reshaped, enforce_sorted=False, batch_first=True)

    out, _ = self.blstm1(x_packed)

    return out

  def trunc_reshape(self, x, x_lens):
    T = x.shape[1]
    if T % 2 != 0:
      x = x[:, :-1, :]
      x_lens = x_lens - 1

    B, T, F = x.shape

    x = torch.reshape(x, (B, T//2, F*2))
    x_lens = torch.clamp(x_lens // 2, min=1)

    return x, x_lens
