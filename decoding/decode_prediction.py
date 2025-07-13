def decode_prediction(output, output_lens, decoder, PHONEME_MAP= LABELS):
  output = output.contiguous()
  output_lens = output_lens.to(torch.int32).contiguous()
  results = decoder(output, output_lens)

  pred_strings = []

  for i in range(output_lens.shape[0]):
    pred_strings.append(" ".join([PHONEME_MAP[token] for token in results[i][0].tokens]))

  return pred_strings

