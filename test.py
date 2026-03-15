import os
import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import AdaIR


class AdaIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            # save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

def print_test_result(results: dict):
    task_num = len(results)
    avg_psnr = avg_ssim = 0.
    print("\n================ Summary ================")
    for task_name, (task_psnr, task_ssim) in results:
        print(f"{task_name:<28} | PSNR: {task_psnr:.2f} | SSIM: {task_ssim:.4f}")
        avg_psnr += task_psnr
        avg_ssim += task_ssim
    avg_psnr, avg_ssim = avg_psnr / task_num, avg_ssim / task_num
    print("------------------------------------------------")
    print(f"Average                      | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=str, default='3task',
                        help='single task: derain, dehaze, deblur, denoise, enhance / all in one: 3task or 5task')
    
    parser.add_argument('--gopro_path', type=str, default="data/test/deblur/", help='save path of test hazy images')
    parser.add_argument('--enhance_path', type=str, default="data/test/enhance/", help='save path of test hazy images')
    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')

    parser.add_argument('--output_path', type=str, default="AdaIR_results/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="adair5d.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    deblur_splits = ["gopro/"]
    enhance_splits = ["lol/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))
    from net.OriSSM_super_better import OriSSM_better
    # net  = AdaIRModel().load_from_checkpoint(ckpt_path).cuda()
    # net.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)  # Lightning ckpt or plain state_dict
    sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    net = OriSSM_better(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 4, 6],
        patch_size=8,
        num_orient=6,
        is_eval=True
    ).cuda()
    net.load_state_dict(sd)
    net.eval()

    test_result = {}

    if testopt.mode == 'denoise':
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

    elif testopt.mode == 'derain':
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

    elif testopt.mode == 'dehaze':
        print('Start testing SOTS...')
        derain_base_path = testopt.derain_path
        name = derain_splits[0]
        testopt.derain_path = os.path.join(derain_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(net, derain_set, task="dehaze")

    elif testopt.mode == 'deblur':
        print('Start testing GOPRO...')
        deblur_base_path = testopt.gopro_path
        name = deblur_splits[0]
        testopt.gopro_path = os.path.join(deblur_base_path,name)
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15, task='deblur')
        test_Derain_Dehaze(net, derain_set, task="deblur")

    elif testopt.mode == 'enhance':
        print('Start testing LOL...')
        enhance_base_path = testopt.enhance_path
        name = derain_splits[0]
        testopt.enhance_path = os.path.join(enhance_base_path,name, task='enhance')
        derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        test_Derain_Dehaze(net, derain_set, task="enhance")

    elif testopt.mode == '3task':
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")

    elif testopt.mode == '5task':
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")

        deblur_base_path = testopt.gopro_path
        for name in deblur_splits:
            print('Start testing GOPRO...')

            # print('Start testing {} rain streak removal...'.format(name))
            testopt.gopro_path = os.path.join(deblur_base_path,name)
            deblur_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='deblur')
            test_Derain_Dehaze(net, deblur_set, task="deblur")

        enhance_base_path = testopt.enhance_path
        for name in enhance_splits:

            print('Start testing LOL...')
            testopt.enhance_path = os.path.join(enhance_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=55, task='enhance')
            test_Derain_Dehaze(net, derain_set, task="enhance")
