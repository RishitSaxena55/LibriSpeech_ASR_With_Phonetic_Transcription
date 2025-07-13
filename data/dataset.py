class AudioDataset(Dataset):
  def __init__(self, root, phonemes):
    self.phonemes = phonemes
    self.mfcc_dir = os.path.join(root, 'mfcc')
    self.transcript_dir = os.path.join(root, 'transcript')

    mfcc_names = sorted(os.listdir(self.mfcc_dir))
    transcript_names = sorted(os.listdir(self.transcript_dir))

    assert len(mfcc_names) == len(transcript_names)

    self.mfccs, self.transcripts = [], []

    for i in range(len(mfcc_names)):
      mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
      mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
      transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))

      if transcript[0] == "[SOS]":
        transcript = transcript[1:]
      if transcript[-1] == "[EOS]":
        transcript = transcript[:-1]

      self.mfccs.append(mfcc)
      self.transcripts.append(transcript)

    self.length = len(self.mfccs)

    phoneme_map = {phoneme: i for i, phoneme in enumerate(phonemes)}

    self.transcripts = [[phoneme_map[p] for p in transcript] for transcript in self.transcripts]

  def __len__(self):
    return self.length

  def __getitem__(self, ind):
    mfcc = torch.FloatTensor(self.mfccs[ind])
    transcript = torch.Tensor(self.transcripts[ind])

    return mfcc, transcript

  def collate_fn(self, batch):
    batched_mfccs, batched_transcripts = zip(*batch)

    mfccs_lens = [mfcc.shape[0] for mfcc in batched_mfccs]
    transcripts_lens = [transcript.shape[0] for transcript in batched_transcripts]

    padded_batched_mfccs = pad_sequence(batched_mfccs, batch_first=True)
    padded_batched_transcripts = pad_sequence(batched_transcripts, batch_first=True)

    time_mask = tat.TimeMasking(time_mask_param=100, iid_masks=True, p=0.8)
    time_masked_padded_batched_mfccs = time_mask(torch.Tensor(padded_batched_mfccs))

    freq_mask = tat.FrequencyMasking(freq_mask_param=4, iid_masks=True)
    time_freq_masked_padded_batched_mfccs = freq_mask(torch.Tensor(time_masked_padded_batched_mfccs))

    return time_freq_masked_padded_batched_mfccs, padded_batched_transcripts, torch.IntTensor(mfccs_lens), torch.IntTensor(transcripts_lens)

class AudioDatasetVal(Dataset):
  def __init__(self, root, phonemes):
    self.phonemes = phonemes
    self.mfcc_dir = os.path.join(root, 'mfcc')
    self.transcript_dir = os.path.join(root, 'transcript')

    mfcc_names = sorted(os.listdir(self.mfcc_dir))
    transcript_names = sorted(os.listdir(self.transcript_dir))

    assert len(mfcc_names) == len(transcript_names)

    self.mfccs, self.transcripts = [], []

    for i in range(len(mfcc_names)):
      mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
      mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
      transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))

      if transcript[0] == "[SOS]":
        transcript = transcript[1:]
      if transcript[-1] == "[EOS]":
        transcript = transcript[:-1]

      self.mfccs.append(mfcc)
      self.transcripts.append(transcript)

    self.length = len(self.mfccs)

    phoneme_map = {phoneme: i for i, phoneme in enumerate(phonemes)}

    self.transcripts = [[phoneme_map[p] for p in transcript] for transcript in self.transcripts]

  def __len__(self):
    return self.length

  def __getitem__(self, ind):
    mfcc = torch.FloatTensor(self.mfccs[ind])
    transcript = torch.Tensor(self.transcripts[ind])

    return mfcc, transcript

  def collate_fn(self, batch):
    batched_mfccs, batched_transcripts = zip(*batch)

    mfccs_lens = [mfcc.shape[0] for mfcc in batched_mfccs]
    transcripts_lens = [transcript.shape[0] for transcript in batched_transcripts]

    padded_batched_mfccs = pad_sequence(batched_mfccs, batch_first=True)
    padded_batched_transcripts = pad_sequence(batched_transcripts, batch_first=True)

    return padded_batched_mfccs, padded_batched_transcripts, torch.IntTensor(mfccs_lens), torch.IntTensor(transcripts_lens)

class AudioDatasetTest(Dataset):
  def __init__(self, root):
    self.mfcc_dir = os.path.join(root, 'mfcc')

    mfcc_names = sorted(os.listdir(self.mfcc_dir))

    self.mfccs = []

    for i in range(len(mfcc_names)):
      mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
      mfcc = (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)
      self.mfccs.append(mfcc)

    self.length = len(self.mfccs)

  def __len__(self):
    return self.length

  def __getitem__(self, ind):
    mfcc = torch.FloatTensor(self.mfccs[ind])
    return mfcc

  def collate_fn(self, batch):
    batched_mfccs = batch

    mfccs_lens = [mfcc.shape[0] for mfcc in batched_mfccs]

    padded_batched_mfccs = pad_sequence(batched_mfccs, batch_first=True)

    return padded_batched_mfccs, torch.IntTensor(mfccs_lens)
