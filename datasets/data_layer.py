import os
import torch
import torch.utils.data as data
import numpy as np

class DataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.uncertainty_dir = args.uncertainty_dir
        self.rgb_feature = args.rgb_feature
        self.flow_feature = args.flow_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.training = phase=='train'

        self.inputs = []
        for session in self.sessions:
            target = np.load(os.path.join(self.data_root, 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([session, start, end, enc_target, dec_target])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                target_matrix[i,j] = target_vector[i+j,:]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]

        rgb_inputs = np.load(os.path.join(self.data_root, self.rgb_feature, session+'.npy'), mmap_mode='r')[start:end]
        rgb_inputs = torch.as_tensor(rgb_inputs.astype(np.float32))
        flow_inputs = np.load(os.path.join(self.data_root, self.flow_feature, session+'.npy'), mmap_mode='r')[start:end]
        flow_inputs = torch.as_tensor(flow_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        return rgb_inputs, flow_inputs, enc_target, dec_target.view(-1, enc_target.shape[-1])

    def __len__(self):
        return len(self.inputs)
