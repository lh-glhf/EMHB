from ..data_provider.data_factory import data_provider
from statsmodels.tsa.arima.model import ARIMA
from ..utils.metrics import metric
import time
import numpy as np
import warnings

class Arima():
    def __init__(self, args):
        args.batch_size = 1
        self.args = args
    
    def fit(self):
        warnings.simplefilter('ignore')
        data_set, data_loader = data_provider(self.args, 'test')
        mae_loss = []
        mse_loss = []
        rmse_loss = []
        mape_loss = []
        mspe_loss = []
        smape_loss = []
        wape_loss = []
        msmape_loss = []
        preds = []
        trues = []
        timer = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            pred_mul = []
            _, seq_len, enc_in = batch_x.shape
            for var in range(enc_in):
                model = ARIMA(batch_x[:, var].squeeze().numpy(), order=(self.args.p, self.args.d, self.args.q), enforce_stationarity=False)
                model_fit = model.fit()
                prediction_var = model_fit.forecast(steps = self.args.pred_len).reshape(-1, 1)
                pred_mul.append(prediction_var)
            predictions = np.concatenate(pred_mul, axis=1)
            predictions = predictions[-self.args.pred_len:, :]
            batch_y = batch_y.squeeze().numpy()[-self.args.pred_len:, :]
            mae, mse, rmse, mape, mspe, smape, wape, msmape = metric(predictions, batch_y)
            mae_loss.append(mae)
            mse_loss.append(mse)
            rmse_loss.append(rmse)
            mape_loss.append(mape)
            mspe_loss.append(mspe)
            smape_loss.append(smape)
            wape_loss.append(wape)
            msmape_loss.append(msmape)
            preds.append(predictions)
            trues.append(batch_y)
            if i % 100 == 0:
                print(f"Step : {i} Time Cost : {time.time() - timer}")
                timer = time.time()
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, smape:{}, wape:{}, msmape: {}'.format(np.average(mse_loss), np.average(mae_loss), np.average(rmse_loss), np.average(mape_loss),
                                                                                                np.average(mspe_loss), np.average(smape_loss), np.average(wape_loss),
                                                                                                np.average(msmape_loss)))
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.model_id,
            self.args.model,
            self.args.data,
            self.args.features,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, ii)
        folder_path = './results/' + setting + '/'
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
