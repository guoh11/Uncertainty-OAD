import argparse
import os
import json

__all__ = ['parse_utrn_args']

def build_data_info(args):
    args.dataset = os.path.basename(os.path.normpath(args.data_root))
    with open(args.data_info, 'r') as f:
        data_info = json.load(f)[args.dataset]
    args.train_session_set = data_info['train_session_set']
    args.test_session_set = data_info['test_session_set']
    args.class_index = data_info['class_index']
    args.num_classes = len(args.class_index)
    return args

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_info', default='data/data_info.json', type=str)
    parser.add_argument('--checkpoint', default='checkpoints/inputs-two-stream-epoch-25.pth', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--verbose', default=True, action='store_true')
    parser.add_argument('--debug', default=True, action='store_true')
    parser.add_argument('--distributed', default=True, action='store_true')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    return parser

def parse_utrn_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/THUMOS', type=str)
    parser.add_argument('--uncertainty_dir', default='uncertainty', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--inputs', default='two-stream', type=str)
    parser.add_argument('--hidden_size', default=4096, type=int)
    parser.add_argument('--pro_size', default=256, type=int)
    parser.add_argument('--rgb_feature', default='resnet200-fc', type=str)
    parser.add_argument('--flow_feature', default='bn_inception', type=str)
    parser.add_argument('--enc_steps', default=64, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    return build_data_info(parser.parse_args())
