import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from net import build_network
from datasets import build_dataset
from utils.config import dict_to_namespace, parse_yaml_opt
from utils.val_utils import AverageMeter, compute_psnr_ssim


def load_network(net, ckpt_path, state_dict_prefix="net."):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    if state_dict_prefix:
        filtered = {}
        for key, value in state_dict.items():
            if key.startswith(state_dict_prefix):
                filtered[key.replace(state_dict_prefix, "", 1)] = value
        if filtered:
            state_dict = filtered

    net.load_state_dict(state_dict, strict=True)
    return net


def test_denoise(net, dataset, sigma=15):
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([_], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            clean_patch = clean_patch.cuda()

            restored = net(degrad_patch)
            temp_psnr, temp_ssim, n = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, n)
            ssim.update(temp_ssim, n)

    print(f"Denoise sigma={sigma}: psnr: {psnr.avg:.2f}, ssim: {ssim.avg:.4f}")
    return [psnr.avg, ssim.avg]


def test_derain_dehaze(net, dataset, task="derain"):
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([_], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            clean_patch = clean_patch.cuda()

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, n = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, n)
            ssim.update(temp_ssim, n)

    print(f"{task} -> PSNR: {psnr.avg:.2f}, SSIM: {ssim.avg:.4f}")
    return [psnr.avg, ssim.avg]


def print_test_result(results: dict, mode: str, ckpt_path: str):
    task_num = len(results)
    avg_psnr = 0.0
    avg_ssim = 0.0
    print("\n================ Summary =====================")
    print(f"model: {ckpt_path} | mode: {mode}")
    print("------------------------------------------------")
    for task_name, (task_psnr, task_ssim) in results.items():
        print(f"{task_name:<28} | PSNR: {task_psnr:.2f} | SSIM: {task_ssim:.4f}")
        avg_psnr += task_psnr
        avg_ssim += task_ssim
    avg_psnr /= task_num
    avg_ssim /= task_num
    print("------------------------------------------------")
    print(f"Average                      | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")


def main():
    opt, opt_path = parse_yaml_opt("AioIR testing")
    print(f"Load option file: {opt_path}")

    np.random.seed(opt.get("seed", 0))
    torch.manual_seed(opt.get("seed", 0))

    cuda_id = opt.get("test", {}).get("cuda", 0)
    torch.cuda.set_device(cuda_id)

    net = build_network(opt["model"]["network_g"]).cuda()
    ckpt_path = opt["path"]["ckpt_path"]
    state_dict_prefix = opt["path"].get("state_dict_prefix", "net.")
    net = load_network(net, ckpt_path, state_dict_prefix=state_dict_prefix)
    net.eval()

    test_result = {}

    test_opt = opt.get("test", {})
    mode = test_opt.get("mode", "5task")

    denoise_base_opt = opt["datasets"]["denoise"]
    denoise_splits = test_opt.get("denoise_splits", ["bsd68/"])
    denoise_tests = []
    for split in denoise_splits:
        dataset_opt = dict(denoise_base_opt)
        dataset_opt["denoise_path"] = os.path.join(denoise_base_opt["denoise_path"], split)
        denoise_tests.append((split, build_dataset(dataset_opt)))

    common_eval_opt = dict_to_namespace(opt["datasets"]["common_eval"])

    if mode == "denoise":
        for split, denoise_set in denoise_tests:
            print(f"Start {split} testing Sigma=15...")
            test_denoise(net, denoise_set, sigma=15)
            print(f"Start {split} testing Sigma=25...")
            test_denoise(net, denoise_set, sigma=25)
            print(f"Start {split} testing Sigma=50...")
            test_denoise(net, denoise_set, sigma=50)
        return

    if mode in ["3task", "5task"]:
        denoise_sigma = test_opt.get("denoise_sigma", [25]) if mode == "5task" else [15, 25, 50]
        for split, denoise_set in denoise_tests:
            for sigma in denoise_sigma:
                print(f"Start {split} testing Sigma={sigma}...")
                test_result[f"denoise-{sigma}"] = test_denoise(net, denoise_set, sigma=sigma)

        derain_splits = test_opt.get("derain_splits", ["Rain100L/"])
        for split in derain_splits:
            common_eval_opt.derain_path = os.path.join(opt["datasets"]["common_eval"]["derain_path"], split)
            derain_set = build_dataset(
                {"type": "DerainDehazeDataset", **vars(common_eval_opt)},
                addnoise=False,
                sigma=55,
                task="derain",
            )
            print(f"Start testing {split} rain streak removal...")
            test_result["derain"] = test_derain_dehaze(net, derain_set, task="derain")
            test_result["dehaze"] = test_derain_dehaze(net, derain_set, task="dehaze")

        if mode == "5task":
            deblur_splits = test_opt.get("deblur_splits", ["gopro/"])
            for split in deblur_splits:
                common_eval_opt.gopro_path = os.path.join(opt["datasets"]["common_eval"]["gopro_path"], split)
                deblur_set = build_dataset(
                    {"type": "DerainDehazeDataset", **vars(common_eval_opt)},
                    addnoise=False,
                    sigma=55,
                    task="deblur",
                )
                print("Start testing GOPRO...")
                test_result["deblur"] = test_derain_dehaze(net, deblur_set, task="deblur")

            enhance_splits = test_opt.get("enhance_splits", ["lol/"])
            for split in enhance_splits:
                common_eval_opt.enhance_path = os.path.join(opt["datasets"]["common_eval"]["enhance_path"], split)
                enhance_set = build_dataset(
                    {"type": "DerainDehazeDataset", **vars(common_eval_opt)},
                    addnoise=False,
                    sigma=55,
                    task="enhance",
                )
                print("Start testing LOL...")
                test_result["enhance"] = test_derain_dehaze(net, enhance_set, task="enhance")

        print_test_result(test_result, mode, ckpt_path)
        return

    if mode in ["derain", "dehaze", "deblur", "enhance"]:
        eval_set = build_dataset(
            {"type": "DerainDehazeDataset", **opt["datasets"]["common_eval"]},
            addnoise=False,
            sigma=15,
            task=mode,
        )
        test_derain_dehaze(net, eval_set, task=mode)
        return

    raise ValueError(f"Unsupported test mode: {mode}")


if __name__ == "__main__":
    main()
