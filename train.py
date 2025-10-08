from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import *

setup_seed(3407)
import argparse
from model.model_new import *
import torch.optim as optim
from data_processor import *
from trainer import *
import time

warnings.filterwarnings("ignore")


def loss_fun(X, Y):
    return F.mse_loss(X, Y)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global args
    args = parser.parse_args()

    ck_file_name = f'rht-light_{args.sensing_rate}.pth'
    if args.model_dim == 32:
        ck_file_name = f'rht_{args.sensing_rate}.pth'

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model_dir = "./%s/%s_ratio_%.2f" % (args.save_dir, args.model, args.sensing_rate)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.backends.cudnn.benchmark = True

    model = CSDemo(ratio=args.sensing_rate, iter_num=args.iter_num, model_dim=args.model_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.flr)

    train_loader, test_loader_bsds, test_loader_set5, test_loader_set14 = data_loader_new(args, is_test=False)

    criterion = loss_fun

    print("Model: %s\nSensing Rate: %.2f\nEpoch: %d\nInitial LR: %f\nParameter: %.0f\nDataset Nums: %.0f\n" % (
        args.model, args.sensing_rate, args.epochs, args.lr, count_parameters(model), len(train_loader.dataset)))

    print('Start training---------------------------------------------------------')

    for epoch in range(args.epochs + 1):
        start_ = time.time()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss = train_new(train_loader, model, criterion, args.sensing_rate, optimizer, device)

        scheduler.step()
        print_data = "[%02d/%02d] Loss: %f" % (epoch, args.epochs, loss)
        print(print_data)
        if epoch > 0 and epoch % 1 == 0:
            torch.save(model.state_dict(), "%s/net_params_%d.pth" % (model_dir, epoch))

            torch.save(model.state_dict(), f"./trained_model/{ck_file_name}")

        if epoch % 1 == 0:
            psnr1, ssim1 = valid_bsds(test_loader_bsds, model, criterion, args.model, False)
            print("BSDS--PSNR: %.2f--SSIM: %.4f" % (psnr1, ssim1))
            psnr2, ssim2 = valid_set(test_loader_set5, model, criterion, args.model, False)
            print("Set5--PSNR: %.2f--SSIM: %.4f" % (psnr2, ssim2))
            psnr3, ssim3 = valid_set(test_loader_set14, model, criterion, args.model, False)
            print("Set14--PSNR: %.2f--SSIM: %.4f" % (psnr3, ssim3))
        print('Running time: {:.2f} seconds'.format(time.time() - start_))
        print()

    print('Trained finished.')


if __name__ == '__main__':
    for cs_ratio in [0.5, 0.4, 0.3, 0.25, 0.1, 0.04]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, default='CSDemo', help='model name')
        parser.add_argument('--sensing_rate', type=float, default=cs_ratio, help='set sensing rate')
        parser.add_argument('--warm_epochs', default=1, type=int, help='number of epochs to warm up')
        parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
        parser.add_argument('-b', '--batch_size', default=16, type=int, help='mini-batch size')
        parser.add_argument('--image-size', default=16 * 6, type=int, metavar='N', help='(default: 96)')
        parser.add_argument('--block_size', default=32, type=int, help='block size')
        parser.add_argument('--lr', '--learning_rate', default=8e-5, type=float, help='initial learning rate')
        parser.add_argument('--flr', '--final_learning_rate', default=3e-6, type=float, help='final learg rate')
        parser.add_argument('--save_dir', help='trained models', default='trained_model', type=str)
        parser.add_argument('--iter_num', type=int, default=10, help='-light is 10, default is 14')
        parser.add_argument('--model_dim', type=int, default=16, help='-light is 16, default is 32')
        main()

