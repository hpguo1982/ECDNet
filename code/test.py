import torch
import argparse
import os
import logging
import sys
import cv2
import time
import utils
from skimage import img_as_ubyte
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion infer")
# Required
# Path save Eval log.
parser.add_argument("--loadDir", type=str, default=' ')
# Path load ck files.
parser.add_argument("--loadDer_cp", type=str, default=' ')
parser.add_argument("--beta_sched", type=str, default='linear', help='cosine or linear')
parser.add_argument("--num_timesteps", type=float, default=500, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl")
parser.add_argument('--size', type=int, default=256, help='test_size')
parser.add_argument('--dataset_name', type=str, default='BUSI', help='test_size')
parser.add_argument('--dataset_root', type=str, default='  ', help='note for this run')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--num_ens', type=int, default=25,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
parser.add_argument('--sampling_timesteps', type=int, default=50,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
parser.add_argument('--cp_condition_net', type=str, default=' ', help='checkpoint for condition network (like PVT)')
# Generation parametersa
parser.add_argument("--device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
parser.add_argument("--self_condition", type=bool, default=True, help="self_condition", required=False)

# Output parameters
parser.add_argument("--print_freq", type=int, default=50, required=False)
parser.add_argument('--job_name', type=str, default='B1S-1DDPM', help='job_name')

args, unparsed = parser.parse_known_args()

# args.job_name = 'DDIM_scale:' + str(args.DDIM_scale) + '_' + 'step_size:' + str(args.step_size)
args.job_name = 'sampling_timesteps:' + str(args.sampling_timesteps) +'_' + 'beta_sched:' + str(args.beta_sched)
args.job_name = args.loadDir + '/results/' + time.strftime("%Y%m%d-%H%M%S-")+ str(args.job_name)
utils.create_exp_dir(args.job_name)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("Infer exp=%s", args.loadDir)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

from metrics import evaluate_single
def main():
    from monai.utils import set_determinism
    set_determinism(1)
    logging.info(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)  # 设置使用第0块GPU
    args.job_name = 'sampling_timesteps:' + str(args.sampling_timesteps) + '_' + 'beta_sched:' + str(
        args.beta_sched)
    args.job_name = args.loadDir + '/results/' + time.strftime("%Y%m%d-%H%M%S-") + str(args.job_name)
    utils.create_exp_dir(args.job_name)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("Infer exp=%s", args.loadDir)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.job_name, "tb"))


    from module.DiffusionModel import DiffSOD

    from Prostate_dataset import Dataset
    # train_dataset = Dataset(args.dataset_root, args.size, 'train', convert_image_to='L')
    test_dataset = Dataset(args.dataset_root, args.size, 'test', convert_image_to='L')
    # batchSize = args.batch_size // args.numSteps

    # sample_batch_size = args.sample_batch_size // args.numSteps

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0,
        pin_memory=True, shuffle=False)

    diffusion = DiffSOD(args, sampling_timesteps=args.sampling_timesteps if args.sampling_timesteps > 0 else None)
    diffusion = diffusion.to(device)


    # 获取检查点文件列表
    checkpoint_dir = args.loadDer_cp
    # checkpoint_files = sorted(os.listdir(checkpoint_dir))
    # checkpoint_files = [f for f in sorted(os.listdir(checkpoint_dir)) if f.endswith('.pt')]
    # num_checkpoints = len(checkpoint_files)
    unsorted_checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    # 定义一个函数来提取文件名中的数字
    import re
    def extract_number(f):
        s = re.findall(r'\d+', f)
        return (int(s[0]) if s else -1, f)
    # 根据提取的数字对文件列表进行排序
    checkpoint_files = sorted(unsorted_checkpoint_files, key=extract_number, reverse=True)
    num_checkpoints = len(checkpoint_files)

    # 前一半的检查点
    checkpoints_to_validate = checkpoint_files[:int(num_checkpoints/2)]

    # 循环验证分配给这个GPU的检查点
    from collections import OrderedDict

    for checkpoint_file in checkpoints_to_validate:
        epoch = checkpoint_file.split("_")[3]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        save_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in save_dict['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # 删除'module.'前缀
            new_state_dict[name] = v

        # 加载状态字典
        diffusion.load_state_dict(new_state_dict, strict=True)

        logging.info('CP_name {}'.format(checkpoint_file))
        score_metricsC = epoch_evaluating(diffusion, checkpoint_file, test_dataloader, device, evaluate_single)
        FG_Dice = score_metricsC["f1"].item() # baseline 80 sota 86
        score_iou_poly = score_metricsC["iou_poly"].item()
        # score_iou_mean = score_metricsC["iou_mean"].item() # baseline 78 sota 82
        score_avg_msd = score_metricsC["avg_msd"].item()  # baseline 2.16 sota 1.96
        score_avg_asd = score_metricsC["avg_asd"].item()  # baseline 2.16 sota 1.96

        logging.info('dataset_name {}'.format(args.dataset_name))
        logging.info('FG_Dice %f', FG_Dice)
        # logging.info('score_f2 %f', score_f2)
        logging.info('Iou_poly %f', score_iou_poly)
        # logging.info('Iou_mean %f', score_iou_mean)
        # logging.info('score_accuracy %f', score_accuracy)
        logging.info('MSD %f', score_avg_msd)
        logging.info('ASD %f', score_avg_asd)

        writer.add_scalars('Metrics', {'FG_Dice_mean': FG_Dice * 10,
                                       'Iou_mean': score_iou_poly * 10,
                                       'Avg_msd': score_avg_msd,
                                       'Avg_asd': score_avg_asd},
                           (int(epoch)))


def epoch_evaluating(model, checkpoint_file, test_dataloader, device, criteria_metrics):
    # Switch model to evaluation mode
    model.eval()
    out_pred_final = torch.FloatTensor().cuda(device)
    out_gt = torch.FloatTensor().cuda(device)  # Tensor stores groundtruth values
    savepath = './prediction/' + args.dataset_name + "/" + checkpoint_file
    with torch.no_grad():  # Turn off gradient
        # For each batch
        test_output_root = os.path.join(args.job_name, savepath)

        for step, (images, masks, index) in enumerate(test_dataloader):
            # Move images, labels to device (GPU)
            input = images.cuda(device)
            masks = masks.cuda(device)
            preds = torch.zeros((input.shape[0], args.num_ens, input.shape[2], input.shape[3])).cuda(device)

            for i in range(args.num_ens):
                sample_output = model.sample(input)
                # Assuming sample_output is a tuple, take the first element
                preds[:, i:i + 1, :, :] = sample_output[0] if isinstance(sample_output, tuple) else sample_output

            preds_mean = preds.mean(dim=1)
            preds_mean[preds_mean < 0.3] = 0

            out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            out_gt = torch.cat((out_gt, masks), 0)

            for idx in range(preds.shape[0]):
                predict_rgb = preds_mean[idx, :, :].cpu().detach()
                predict_rgb = img_as_ubyte(predict_rgb)
                if not os.path.exists(test_output_root):
                    os.makedirs(test_output_root)
                cv2.imwrite(os.path.join(test_output_root, f"{index[idx]}.png"), predict_rgb)

            if step % args.print_freq == 0 or step == len(test_dataloader) - 1:
                logging.info("val: Step {:03d}/{:03d}".format(step, len(test_dataloader) - 1))

    _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
        out_pred_final, out_gt)

    score_metricsC = {
        "recall": _recallC,
        "specificity": _specificityC,
        "precision": _precisionC,
        "f1": _F1C,
        "f2": _F2C,
        "accuracy": _ACC_overallC,
        "iou_poly": _IoU_polyC,
        "iou_bg": _IoU_bgC,
        "iou_mean": _IoU_meanC,
        "avg_msd": _MSD,
        "avg_asd": _ASD,
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score_metricsC
if __name__ == '__main__':
    main()
