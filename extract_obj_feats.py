import argparse
import os
import os.path as osp
import time
import warnings
import copy

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import get_root_logger, multi_gpu_test, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import (
    build_ddp,
    build_dp,
    compat_cfg,
    get_device,
    replace_cfg_vals,
    setup_multi_processes,
    update_data_root,
)

from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--data_split",
        default="train",
        help="the dataset train, val, test to load from cfg file",
    )
    parser.add_argument(
        "--fg_iou_thr", default=0.6, help="fg iou thr > to be extracted only "
    )
    parser.add_argument(
        "--bg_iou_thr", default=0.3, help="bg iou thr < to be extracted only"
    )

    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed testing)",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both "
            "specified, --options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def extract_feats(
    model, dataset, fg_th, bg_th, save_dir, data_split="train", logger=None
):
    model.eval()
    results = []

    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        # image = item["img"].data[None, :]
        bbox_feats, bbox_labels, bboxes = model.feats_extract(
            item["img"].data[None, :],
            item["img_metas"].data,
            item["gt_bboxes"].data,
            item["gt_labels"].data,
        )
        # logger.info(
        #     f"{index:05}/{len(data_loaders[0])} feats shape - {bbox_feats.shape}"
        # )

        feats.append(bbox_feats.data.cpu().numpy())
        labels.append(bbox_labels.data.cpu().numpy())
        del data, bbox_feats, bbox_labels, bboxes

        for _ in range(batch_size):
            prog_bar.update()

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    split = f"{fg_th}_{bg_th}"

    np.save(f"{save_dir}/{data_split}_{split}_feats.npy", feats)
    np.save(f"{save_dir}/{data_split}_{split}_labels.npy", labels)
    # import pdb; pdb.set_trace()
    print(f"{labels.shape} num of features")

    return results


def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    cfg = Config.fromfile(args.config)

    print("cfg", cfg.model.train_cfg)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    elif "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get("neck"):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get("rfp_backbone"):
            if cfg.model.neck.rfp_backbone.get("pretrained"):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed testing. Use the first GPU "
            "in `gpu_ids` now."
        )
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False
    )

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get("test_dataloader", {}),
    }

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        json_file = osp.join(args.work_dir, f"eval_{timestamp}.json")

    logger = get_root_logger(cfg.log_level)

    cfg.model.train_cfg.rcnn.assigner.pos_iou_thr = args.fg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.min_pos_iou = args.fg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.neg_iou_thr = args.bg_iou_thr

    fg_th = cfg.model.train_cfg.rcnn.assigner.pos_iou_thr
    bg_th = cfg.model.train_cfg.rcnn.assigner.neg_iou_thr

    # build the dataloader
    dataset = build_dataset(cfg.data.train)

    # data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    # cfg.model.train_cfg = None
    # model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # fp16_cfg = cfg.get("fp16", None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # if "CLASSES" in checkpoint.get("meta", {}):
    #     model.CLASSES = checkpoint["meta"]["CLASSES"]
    # else:

    if distributed:
        print("script does not work with distributed")

    # model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.cuda()
    model.CLASSES = dataset.CLASSES
    # exit()
    extract_feats(
        model, dataset, fg_th, bg_th, args.work_dir, args.data_split, logger=logger
    )


if __name__ == "__main__":
    main()
