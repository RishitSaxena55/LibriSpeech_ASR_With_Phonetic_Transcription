def calculate_levenshtein(output, label, output_lens, label_lens, decoder, PHONEME_MAP=LABELS):
  dist = 0
  batch_size = label.shape[0]

  pred_strings = decode_prediction(output, output_lens, decoder, PHONEME_MAP)

  for i in range(batch_size):
    pred_string = pred_strings[i]
    label_string = " ".join([PHONEME_MAP[int(token)] for token in label[i][:int(label_lens[i])].tolist()])

    dist += Levenshtein.distance(pred_string, label_string)

  dist /= batch_size

  return dist
