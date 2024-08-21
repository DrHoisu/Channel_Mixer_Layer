import os
import re
import torch
from openstl.utils import create_parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    print('>'*35 + ' merging value and index prediction networks ' + '<'*35)
    base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
    exp_dir1, exp_dir2 = args.ex_name.split(',')
    assert exp_dir1 is not None and exp_dir2 is not None, "two directories of experiment folders are required for testing"

    if not exp_dir1.startswith(args.res_dir):
        ckpt_dir1 = os.path.join(base_dir, exp_dir1)
    if not exp_dir2.startswith(args.res_dir):
        ckpt_dir2 = os.path.join(base_dir, exp_dir2)
    best_model_path1 = os.path.join(ckpt_dir1, 'checkpoint.pth')
    best_model_path2 = os.path.join(ckpt_dir2, 'checkpoint.pth')
    model_checkpoint1 = torch.load(best_model_path1)
    model_checkpoint2 = torch.load(best_model_path2)
    print(best_model_path1, best_model_path2)

    state_dict1 = model_checkpoint1
    state_dict2 = model_checkpoint2

    merge_dict = dict(state_dict1, **state_dict2)
    exp_dir_name1, exp_dir_name2 = exp_dir1.split('/')[-1], exp_dir2.split('/')[-1]
    merge_dict_dir_name = exp_dir_name1 + '_' + exp_dir_name2
    exp_dir_path = exp_dir1.split(exp_dir_name1)[0]
    merge_dict_path = os.path.join(base_dir, exp_dir_path, merge_dict_dir_name)
    print('merged model folder: ' + merge_dict_path)
    if not os.path.exists(merge_dict_path):
        os.makedirs(merge_dict_path)
    merge_checkpoint_path = os.path.join(merge_dict_path, 'checkpoint.pth')
    torch.save(merge_dict, merge_checkpoint_path)
    print('>'*50 + ' merge success' + '<'*50)

