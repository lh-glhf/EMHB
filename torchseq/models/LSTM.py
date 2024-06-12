import torch
import torch.nn as nn
import os


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size = args.seq_len
        self.args = args
        self.hidden_layer_size = args.hidden_layer_size
        self.num_layers = args.num_layers
        self.device = self._acquire_device()
        output_size = args.pred_len
        num_features = args.enc_in
        super().__init__()
        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(self.num_layers, num_features, self.hidden_layer_size).to(self.device),
                            torch.zeros(self.num_layers, num_features, self.hidden_layer_size).to(self.device))

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def forward(self, x, x_mark, dec_inp, y_mark):
        B, L, D = x.shape
        lstm_out, self.hidden_cell = self.lstm(x.view(B, D, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.detach().view(-1, self.hidden_layer_size))
        return predictions.view(B, -1, D)


