import sys, os
from train import *


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="linear",
                        help="Linear or Conv GANS ['linear' / 'conv']")
    parser.add_argument("--bn", default='batch',
                        help="Applying Batch normalization to the model or not ['nobatch' / 'batch' / 'spectral']")
    parser.add_argument("--device", default="cpu",
                        help="Device accelerator ['cpu' / 'cuda']")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()
    args.model
    if args.bn.lower() =='batch':
        batch_norm = True
    if args.bn.lower() == 'nobatch':
        batch_norm = False
    if args.bn.lower() == 'spectral':
        spectral_norm=True


    if args.model.lower() =='linear':
        print("Linear GAN")
        train_loop = Train(is_linear=True, batch_norm=batch_norm, 
            spectral_norm=False, z_dim=64, device=args.device)
        train_loop.trainer(lr= 0.00001, n_epochs=100)

    elif args.model.lower() =='conv':
        print("DCGAN")
        train_loop = Train(is_linear=False, batch_norm=True, 
                   spectral_norm=spectral_norm, z_dim=128, device=args.device)
        train_loop.trainer(lr= 0.0002)
