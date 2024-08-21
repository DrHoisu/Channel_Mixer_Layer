import os
import re
import copy
import torch
from openstl.utils import create_parser

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    print('>'*35 + ' converting value to index prediction networks ' + '<'*35)
    base_dir = args.res_dir if args.res_dir is not None else 'work_dirs'
    exp_dir = args.ex_name
    assert exp_dir is not None, "experiment folder is required"

    if not exp_dir.startswith(args.res_dir):
        ckpt_dir = os.path.join(base_dir, exp_dir)
    best_model_path = os.path.join(ckpt_dir, 'checkpoint.pth')

    model_checkpoint = torch.load(best_model_path)
    #print(best_model_path1, best_model_path2)
    state_dict = model_checkpoint #['state_dict']
    #print(state_dict.keys())

    parameter = copy.deepcopy(state_dict)
    for name, param in state_dict.items():
        new_name = 'index_net.' + name
        parameter.update({new_name:param})
        parameter.pop(name)

    converted_path = os.path.join(ckpt_dir, 'modeltoindex.pth')
    torch.save(parameter, converted_path)
    #print(parameter.keys())
    print('convert model folder: ' + converted_path)
    print('>'*50 + ' convert success' + '<'*50)
