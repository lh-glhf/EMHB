from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS, Dataset_M4, PSMSegLoader,\
    MSLSegLoader, SMAPSegLoader,SWATSegLoader, SMDSegLoader, UEAloader, Dataset_ETT_hour_TimesNet, Dataset_Custom_TimesNet, Dataset_ETT_minute_TimesNet
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_TimesNet': Dataset_ETT_hour_TimesNet,
    'ETTh2': Dataset_ETT_hour,
    'ETTh2_TimesNet': Dataset_ETT_hour_TimesNet,
    'ETTm1': Dataset_ETT_minute,
    'ETTm1_TimesNet': Dataset_ETT_minute_TimesNet,
    'ETTm2': Dataset_ETT_minute,
    'ETTm2_TimesNet': Dataset_ETT_minute_TimesNet,
    'custom': Dataset_Custom,
    'custom_TimesNet': Dataset_Custom_TimesNet,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'm4_TimesNet': Dataset_M4,
    'PSM_TimesNet': PSMSegLoader,
    'MSL_TimesNet': MSLSegLoader,
    'SMAP_TimesNet': SMAPSegLoader,
    'SMD_TimesNet': SMDSegLoader,
    'SWAT_TimesNet': SWATSegLoader,
    'UEA_TimesNet': UEAloader
}


def data_provider(args, flag):
    if args.model == "TimesNet":
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        shuffle_flag = False if flag == 'test' else True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader


