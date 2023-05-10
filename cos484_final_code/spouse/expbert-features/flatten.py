import torch
dat = torch.load(f'dev_exp_features.bin')
dat = dat.reshape(-1, 41, 768)
torch.save(dat, f'dev.bin')
dat = torch.load(f'test_exp_features.bin')
dat = dat.reshape(-1, 41, 768)
torch.save(dat, f'test.bin')
