from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom

def data_provider(args, flag):
    dataset = Dataset_Custom(args, args.root_path, flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        data_path=args.data_path, target=args.target,
        timeenc=1 if args.embed == 'timeF' else 0, freq=args.freq)
    if dataset.blocks.shape[-1] != args.enc_in:
        raise ValueError(f'CSV supplies {dataset.blocks.shape[-1]} channels, but enc_in={args.enc_in}')
    print(flag, len(dataset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=flag == 'train',
                        num_workers=args.num_workers, drop_last=False)
    return dataset, loader
