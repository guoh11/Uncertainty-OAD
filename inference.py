import os
import time

import torch
import numpy as np
import utils as utl
import configs
import models

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_score_metrics = []
    enc_target_metrics = []
    dec_score_metrics = [[] for i in range(args.dec_steps)]
    dec_target_metrics = [[] for i in range(args.dec_steps)]

    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = models.UTRN(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    softmax = torch.nn.Softmax(dim=1).to(device)

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            rgb_inputs = np.load(os.path.join(args.data_root, args.rgb_feature, session+'.npy'), mmap_mode='r')
            flow_inputs = np.load(os.path.join(args.data_root, args.flow_feature, session+'.npy'), mmap_mode='r')
            target = np.load(os.path.join(args.data_root, 'target', session+'.npy'))
            future_input = to_device(torch.zeros(model.future_size), device)
            enc_hidden = to_device(torch.zeros(model.hidden_size), device)
            enc_cell = to_device(torch.zeros(model.hidden_size), device)

            for l in range(target.shape[0]):
                rgb_input = to_device(torch.as_tensor(rgb_inputs[l].astype(np.float32)), device)
                flow_input = to_device(torch.as_tensor(flow_inputs[l].astype(np.float32)), device)

                future_input, enc_hidden, enc_cell, enc_score, dec_score_stack = model.step(rgb_input, flow_input, future_input, enc_hidden, enc_cell)

                enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])
                enc_target_metrics.append(target[l])

                for step in range(args.dec_steps):
                    dec_score_metrics[step].append(softmax(dec_score_stack[step]).cpu().numpy()[0])
                    dec_target_metrics[step].append(target[min(l + step, target.shape[0] - 1)])
        end = time.time()

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(
            session, session_idx, len(args.test_session_set), end - start))

    save_dir = os.path.dirname(args.checkpoint)
    result_file  = os.path.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder
    utl.compute_result_multilabel(args.class_index,
                                  enc_score_metrics, enc_target_metrics,
                                  save_dir, result_file, ignore_class=[21], save=True, verbose=True)

    # Compute result for decoder
    for step in range(args.dec_steps):
        utl.compute_result_multilabel(args.class_index,
                                      dec_score_metrics[step], dec_target_metrics[step],
                                      save_dir, result_file, ignore_class=[21], save=False, verbose=True)

if __name__ == '__main__':
    main(configs.parse_utrn_args())
