import argparse
import yaml
import torch
import random
import numpy as np
from .data_provider.data_config import data_collection, data_split, data_testStamp
from .utils.tools import string_split
from .exps.exp_FEDformer import Exp_FEDformer
from .exps.exp_Autoformer import Exp_Autoformer
from .exps.exp_Informer import Exp_Informer
from .exps.exp_ModernTCN import Exp_ModernTCN
from .exps.exp_PatchTST import Exp_PatchTST
from .exps.exp_Crossformer import Exp_crossformer
from .exps.exp_iTransformer import Exp_iTransformer
from .exps.exp_Linear import Exp_Linear
from .exps.exp_MICN import Exp_MICN
from .exps.exp_TimesNet import Exp_TimesNet
from .exps.exp_LSTM import Exp_LSTM
from .exps.exp_RNN import Exp_RNN
from .exps.exp_SCINet import Exp_SCINet
from .exps.exp_Nbeats import Exp_Nbeats
from .exps.exp_Arima import Exp_Arima
model_entrypoints = {
    'Informer': Exp_Informer,
    'Autoformer': Exp_Autoformer,
    'FEDformer': Exp_FEDformer,
    'ModernTCN': Exp_ModernTCN,
    'PatchTST': Exp_PatchTST,
    'Crossformer': Exp_crossformer,
    'iTransformer': Exp_iTransformer,
    'iInformer': Exp_iTransformer,
    'iReformer': Exp_iTransformer,
    'iFlowformer': Exp_iTransformer,
    'NLinear': Exp_Linear,
    'DLinear': Exp_Linear,
    'MICN': Exp_MICN,
    'TimesNet': Exp_TimesNet,
    'LSTM': Exp_LSTM,
    'RNN': Exp_RNN,
    'SCINet': Exp_SCINet,
    'Nbeats': Exp_Nbeats,
    'Arima': Exp_Arima,
}

Training_Models = ['Informer', 'Autoformer', 'FEDformer', 'ModernTCN', 'PatchTST', 'Crossformer', 'iTransformer', 'iInformer', 'iReformer', 'iFlowformer', 'NLinear', 'DLinear', 'MICN',
                   'TimesNet', 'LSTM', 'RNN', 'SCINet', 'Nbeats']
UniqueData_Models = ['Crossformer']  # TimesNet is Special in Coding
class TorchSeqModel():
    def __init__(self, exp, args):
        self.exp = exp
        # self.model = None
        self.args = args
        self.setting = "Setting not set"
        if self.args.model in Training_Models:
            # self.model = exp._get_model()
            self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                self.args.model, self.args.data, self.args.features,
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff,
                self.args.attn, self.args.factor,
                self.args.embed, self.args.distil, self.args.mix, self.args.des, 0)

    def train(self, itr=0):
        if self.args.model in Training_Models:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
            self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                self.args.model, self.args.data, self.args.features,
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff,
                self.args.attn, self.args.factor,
                self.args.embed, self.args.distil, self.args.mix, self.args.des, itr)
            self.exp.train(self.setting)
        elif self.args.model in ['Arima']:
            self.exp.train()

    def test(self, itr=0):
        if self.args.model in Training_Models:
            print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
            self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                self.args.model, self.args.data, self.args.features,
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.e_layers, self.args.d_layers, self.args.d_ff,
                self.args.attn, self.args.factor,
                self.args.embed, self.args.distil, self.args.mix, self.args.des, itr)
            self.exp.test(self.setting)
        elif self.args.model in ['Arima']:
            self.exp.test()

    def load(self):
        if self.args.ckpt_path:
            if self.args.model not in ['Nbeatsx', 'DTRD']:
                print('>>>Loading from : {}>>'.format(self.args.ckpt_path))
                self.exp.model.load_state_dict(torch.load(self.args.ckpt_path))
                print('>>>>>>>Load Succeed!>>>>>>>>>>>>>>>>>>>>>>>>>>')
                # ms.save_checkpoint(self.exp.model, "./checkpoints/test_ckpt/ALLOT_PEMS08.ckpt")
        else:
            raise RuntimeError('Need ckpt_path but get None')


def is_model(model_name):
    return model_name in model_entrypoints.keys()


def create_model(
        model_name: str,
        data_name: str,
        pretrained: bool = False,
        checkpoint_path: str = '',
        config_file: str = '',
        **kwargs,
):
    if checkpoint_path == "" and pretrained:
        raise ValueError("checkpoint_path is mutually exclusive with pretrained")

    # 创建一个命名空间对象
    args = argparse.Namespace(
        model_name=model_name,
        data_name=data_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        **kwargs  # 添加 **kwargs 中的参数
    )
    if args.config_file:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            args.__dict__.update(config_data)
    if hasattr(args, 'model'):
        args.model = model_name
    if hasattr(args, 'data'):
        args.data = data_name
    if args.model not in UniqueData_Models:
        data_parser = data_collection
        if hasattr(args, 'data'):
            if args.data in data_parser.keys():
                data_info = None
                if hasattr(args, 'data'):
                    data_info = data_parser[args.data]
                args.data = data_info['dataset']
                if args.model == 'TimesNet':
                    args.data = args.data + "_TimesNet"
                if hasattr(args, 'data_path') and hasattr(args, 'target'):
                    args.data_path = data_info['data_provider']
                    args.target = data_info['T']
                if hasattr(args, 'enc_in') and hasattr(args, 'dec_in') and hasattr(args, 'c_out') and hasattr(args,
                                                                                                              'features'):
                    args.enc_in, args.dec_in, args.c_out = data_info[args.features]
    elif args.model == 'Crossformer':
        data_parser = data_collection
        if args.data in data_parser.keys():
            data_info = data_parser[args.data]
            args.data_path = data_info['data_provider']
            args.data_dim = data_info['M'][0]
        args.in_len = args.seq_len
        args.out_len = args.pred_len
    if args.model == 'MICN':
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in args.conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((args.seq_len + args.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((args.seq_len + args.pred_len + ii - 1) // ii)
        args.isometric_kernel = isometric_kernel  # kernel of isometric convolution
        args.decomp_kernel = decomp_kernel  # kernel of decomposition operation
    if args.model == "SCINet":
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]
    print('Args in experiment:')
    print(args)

    # set mode
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    if hasattr(args, "seed"):
        fix_seed= args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

    # Check model_name
    if not is_model(args.model_name):
        raise RuntimeError(f'Unknow model {args.model_name}, options:{model_entrypoints.keys()}')

    exp = model_entrypoints[args.model_name](args)
    Exp = TorchSeqModel(exp, args)

    if args.pretrained == True and args.checkpoint_path != '':
        Exp.load()

    return Exp

    # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
    #             args.model, args.data_provider, args.features,
    #             args.seq_len, args.label_len, args.pred_len,
    #             args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
    #             args.embed, args.distil, args.mix, args.des, 0)
    # print("Setting:", setting)
    # exp = model_entrypoints[args.model_name](args)
    # model = exp._get_model()
    # exp.train(model, setting)

    # if pretrained and checkpoint_path!="":
    #     pass
    # return args


if __name__ == '__main__':
    print(create_model('Informer', 'ETTh1', pretrained=False, config_file='../../configs/Informer/informer_GPU.yaml'))
