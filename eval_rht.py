import os
import warnings
from skimage.metrics import structural_similarity as ssim_count
from skimage.metrics import peak_signal_noise_ratio as psnr_count
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from model.model_new import *

warnings.filterwarnings("ignore")

model_rht = None
old_rate_rht = 10000
old_flops_rht = None


def eval_rht_api(input_np, cs_ratio, inputs_clean_np=None, ROOT_PATH='../RHTNet/', use_light=False):
    global old_rate_rht, model_rht, old_flops_rht

    iter_num = 14 if not use_light else 10
    model_dim = 32 if not use_light else 16

    if model_rht is None or (not cs_ratio == old_rate_rht):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_rht = CSDemo(ratio=cs_ratio, iter_num=iter_num, model_dim=model_dim)
        model_rht = model_rht.cuda()
        model_dir = f"{ROOT_PATH}trained_model"
        checkpoint = torch.load(f"{model_dir}/rht{'-light' if use_light else ''}_{str(cs_ratio)}.pth",
                                map_location=device)

        model_rht.load_state_dict(checkpoint, strict=True)
        model_rht.eval()

    if inputs_clean_np is not None:
        inputs_clean_np = inputs_clean_np.copy()
    else:
        inputs_clean_np = input_np.copy()

    input_x = torch.from_numpy(input_np.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()

    import time
    start = time.time()
    x_output, _ = model_rht(input_x)
    end = time.time()

    if (not cs_ratio == old_rate_rht):
        import logging
        from fvcore.nn import FlopCountAnalysis
        logging.getLogger().setLevel(logging.ERROR)
        flops = FlopCountAnalysis(model_rht, input_x).total()
        old_flops_rht = flops
    else:
        flops = old_flops_rht

    params = sum(p.numel() for p in model_rht.parameters() if p.requires_grad)

    x_output = x_output.squeeze(0).squeeze(0)
    X_rec = x_output.cpu().data.numpy()

    rec_PSNR = psnr_count(X_rec, inputs_clean_np, data_range=1)
    rec_SSIM = ssim_count(X_rec, inputs_clean_np, data_range=1)

    old_rate_rht = cs_ratio
    return X_rec, rec_PSNR, rec_SSIM, end - start, params, flops


def rgb_to_ycbcr(input):
    # input is mini-batch N x 3 x H x W of an RGB image
    output = Variable(input.data.new(*input.size()))
    output[:, 0, :, :] = input[:, 0, :, :] * 65.481 + input[:, 1, :, :] * 128.553 + input[:, 2, :, :] * 24.966 + 16
    # similarly write output[:, 1, :, :] and output[:, 2, :, :] using formulas from https://en.wikipedia.org/wiki/YCbCr
    return output


if __name__ == "__main__":
    image_dir = './dataset/set5/set5'
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampling_rate = 0.1
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])

    avg_psnr = 0
    avg_ssim = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        x_channel = rgb_to_ycbcr(image_tensor)[:, 0, :, :] / 255.
        x_np = x_channel.squeeze().numpy()
        model_rec, model_psnr, model_ssim, model_time, model_parameter, model_flops = eval_rht_api(
            input_np=x_np,
            cs_ratio=sampling_rate,
            inputs_clean_np=x_np,
            use_light=True
        )
        avg_psnr += model_psnr
        avg_ssim += model_ssim

    print(f"Avg-PSNR: {avg_psnr / len(image_files)}")
    print(f"Avg-SSIM: {avg_ssim / len(image_files)}")
