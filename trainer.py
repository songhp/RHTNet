from utils import *

from skimage.metrics import structural_similarity as ssim
from torch.cuda.amp import autocast, GradScaler


def train_new(train_loader, model, criterion, sensing_rate, optimizer, device):
    model.train()
    sum_loss = 0
    scaler = GradScaler()

    for inputs, _ in train_loader:
        inputs = inputs[:, 0:1, :, :].to(device)
        optimizer.zero_grad()

        with autocast():
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.item()

    return sum_loss


def valid_bsds(valid_loader, model, criterion, model_name='csdemo', is_test=False):
    sum_psnr = 0
    sum_ssim = 0
    device = test_device
    model.eval()
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr

            sum_ssim += ssim(outputs[0].squeeze().cpu().numpy(), inputs.squeeze().cpu().numpy(), data_range=1)

    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)


def valid_set(valid_loader, model, criterion, model_name='csdemo', is_test=False):
    sum_psnr = 0
    sum_ssim = 0
    device = test_device
    model.eval()
    time_sum = 0
    with torch.no_grad():
        for iters, (inputs, _) in enumerate(valid_loader):
            inputs = rgb_to_ycbcr(inputs.to(device))[:, 0, :, :].unsqueeze(1) / 255.
            outputs = model(inputs)
            mse = F.mse_loss(outputs[0], inputs)
            psnr = 10 * log10(1 / mse.item())
            sum_psnr += psnr
            sum_ssim += ssim(outputs[0].squeeze().cpu().numpy(), inputs.squeeze().cpu().numpy(), data_range=1)
    return sum_psnr / len(valid_loader), sum_ssim / len(valid_loader)
