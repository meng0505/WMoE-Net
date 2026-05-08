

"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""



import json

import time

from pathlib import Path



import numpy as np

import torch



from ultralytics.cfg import get_cfg, get_save_dir

from ultralytics.data.utils import check_cls_dataset, check_det_dataset

from ultralytics.nn.autobackend import AutoBackend

from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis

from ultralytics.utils.checks import check_imgsz

from ultralytics.utils.ops import Profile

from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode

from ultralytics.nn.modules.moe import MoEFusionWrapper

from ultralytics.utils import LOGGER

from ultralytics.utils.metrics import  ap_by_size_with_xs, ap_by_size, compute_gt_area_all





def _iter_moe_modules(model):

    """
    统一遍历方式：
    - 训练过程内的 val：model 是 nn.Module，直接 model.modules()
    - 独立 validate（AutoBackend）：仅当是 PyTorch 权重才有 model.model.modules()
    """

    try:

        yield from model.modules()

    except AttributeError:

        if hasattr(model, "pt") and model.pt and hasattr(model, "model") and model.model is not None:

            yield from model.model.modules()

        else:

            if hasattr(model, "model") and model.model is not None:

                yield from ()

            else:

                yield from ()



class BaseValidator:

    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """



    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):

        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """

        self.args = get_cfg(overrides=args)

        self.dataloader = dataloader

        self.pbar = pbar

        self.stride = None

        self.data = None

        self.device = None

        self.batch_i = None

        self.training = True

        self.names = None

        self.seen = None

        self.stats = None

        self.confusion_matrix = None

        self.nc = None

        self.iouv = None

        self.jdict = None

        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}



        self.save_dir = save_dir or get_save_dir(self.args)

        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        if self.args.conf is None:

            self.args.conf = 0.001

        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)



        self.plots = {}

        self.callbacks = _callbacks or callbacks.get_default_callbacks()









    def _moe_load_gates(self, model, gates_path: str):

        """
        模型就绪后调用一次：把 gates.json 只加载到“开启先验”的 MoE 上；
        若所有 MoE 都关闭先验（prior_alpha<=0），则不加载也不打印“Loaded...”日志。
        """



        moe_list = [m for m in _iter_moe_modules(model) if isinstance(m, MoEFusionWrapper)]

        if not moe_list:

            LOGGER.warning("[MoE] No MoEFusionWrapper found; skip priors.")

            return

        moe_with_prior = [m for m in moe_list if not getattr(m, "no_prior", False)]

        if not moe_with_prior:



            LOGGER.info("[MoE] All MoE modules run WITHOUT priors (prior_alpha<=0).")

            return





        try:

            with open(gates_path, "r", encoding="utf-8") as f:

                gates = json.load(f)

        except FileNotFoundError:

            LOGGER.warning(f"[MoE] gates.json not found: {gates_path} -> running WITHOUT priors.")

            return

        except Exception as e:

            LOGGER.warning(f"[MoE] Failed to load gates.json: {e} -> running WITHOUT priors.")

            return





        for m in moe_with_prior:

            m.load_gates(gates)

        LOGGER.info(f"[MoE] Loaded gates.json into {len(moe_with_prior)} MoEFusionWrapper modules from: {gates_path}")



    def _moe_set_batch_stems(self, model, batch):

        """
        每个 batch 推理前：把本批图名 stem 挂到每个“开启先验”的 MoE 上；
        关先验时跳过（减少无用属性写入）。
        """

        try:

            files = batch.get("im_file", []) if isinstance(batch, dict) else []

            from pathlib import Path

            stems = [Path(p).stem for p in files]

            meta = {"stems": stems}



            for m in _iter_moe_modules(model):

                if isinstance(m, MoEFusionWrapper) and not getattr(m, "no_prior", False):

                    m._current_batch_meta = meta

        except Exception as e:

            LOGGER.warning(f"[MoE] set batch stems failed: {e}")



    @smart_inference_mode()

    def __call__(self, trainer=None, model=None):

        """Executes validation process, running inference on dataloader and computing performance metrics."""

        self.training = trainer is not None

        augment = self.args.augment and (not self.training)

        if self.training:

            self.device = trainer.device

            self.data = trainer.data



            self.args.half = self.device.type != "cpu" and trainer.amp

            model = trainer.ema.ema or trainer.model

            model = model.half() if self.args.half else model.float()



            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)

            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)

            model.eval()

            gates_path = getattr(self.args, "moe_gates", None)

            if gates_path:

                self._moe_load_gates(model, gates_path)



        else:

            if str(self.args.model).endswith(".yaml") and model is None:

                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")

            callbacks.add_integration_callbacks(self)

            model = AutoBackend(

                weights=model or self.args.model,

                device=select_device(self.args.device, self.args.batch),

                dnn=self.args.dnn,

                data=self.args.data,

                fp16=self.args.half,

            )



            self.device = model.device

            self.args.half = model.fp16

            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

            imgsz = check_imgsz(self.args.imgsz, stride=stride)

            if engine:

                self.args.batch = model.batch_size

            elif not pt and not jit:

                self.args.batch = model.metadata.get("batch", 1)

                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")



            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:

                self.data = check_det_dataset(self.args.data)

            elif self.args.task == "classify":

                self.data = check_cls_dataset(self.args.data, split=self.args.split)

            else:

                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))



            if self.device.type in {"cpu", "mps"}:

                self.args.workers = 0

            if not pt:

                self.args.rect = False

            self.stride = model.stride

            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)



            model.eval()



            try:

                model.warmup(imgsz=(1 if pt else self.args.batch, self.args.ch, imgsz, imgsz))

            except Exception as e:



                if getattr(self.args, "verbose", True):

                    print(f"[val] warmup skipped: {e}")







        self.run_callbacks("on_val_start")

        dt = (

            Profile(device=self.device),

            Profile(device=self.device),

            Profile(device=self.device),

            Profile(device=self.device),

        )

        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))

        self.init_metrics(de_parallel(model))

        self.jdict = []

        for batch_i, batch in enumerate(bar):

            self.run_callbacks("on_val_batch_start")

            self.batch_i = batch_i



            with dt[0]:

                batch = self.preprocess(batch)





            with dt[1]:





                self._moe_set_batch_stems(model, batch)



                preds = model(batch["img"], augment=augment)





            with dt[2]:

                if self.training:

                    self.loss += model.loss(batch, preds)[1]





            with dt[3]:

                preds = self.postprocess(preds)



            self.update_metrics(preds, batch)

            if self.args.plots and batch_i < 3:

                self.plot_val_samples(batch, batch_i)

                self.plot_predictions(batch, preds, batch_i)



            self.run_callbacks("on_val_batch_end")

        stats = self.get_stats()

        self.check_stats(stats)

        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))

        self.finalize_metrics()

        self.print_results()

        self.run_callbacks("on_val_end")

        if self.training:

            model.float()

            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}

            return {k: round(float(v), 5) for k, v in results.items()}

        else:

            LOGGER.info(

                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(

                    *tuple(self.speed.values())

                )

            )

            if self.args.save_json and self.jdict:

                with open(str(self.save_dir / "predictions.json"), "w") as f:

                    LOGGER.info(f"Saving {f.name}...")

                    json.dump(self.jdict, f)

                stats = self.eval_json(stats)

            if self.args.plots or self.args.save_json:

                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

            return stats



    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):

        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """



        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)



        correct_class = true_classes[:, None] == pred_classes

        iou = iou * correct_class

        iou = iou.cpu().numpy()

        for i, threshold in enumerate(self.iouv.cpu().tolist()):

            if use_scipy:



                import scipy



                cost_matrix = iou * (iou >= threshold)

                if cost_matrix.any():

                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)

                    valid = cost_matrix[labels_idx, detections_idx] > 0

                    if valid.any():

                        correct[detections_idx[valid], i] = True

            else:

                matches = np.nonzero(iou >= threshold)

                matches = np.array(matches).T

                if matches.shape[0]:

                    if matches.shape[0] > 1:

                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]

                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]



                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                    correct[matches[:, 1].astype(int), i] = True

        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)



    def add_callback(self, event: str, callback):

        """Appends the given callback."""

        self.callbacks[event].append(callback)



    def run_callbacks(self, event: str):

        """Runs all callbacks associated with a specified event."""

        for callback in self.callbacks.get(event, []):

            callback(self)



    def get_dataloader(self, dataset_path, batch_size):

        """Get data loader from dataset path and batch size."""

        raise NotImplementedError("get_dataloader function not implemented for this validator")



    def build_dataset(self, img_path):

        """Build dataset."""

        raise NotImplementedError("build_dataset function not implemented in validator")



    def preprocess(self, batch):

        """Preprocesses an input batch."""

        return batch



    def postprocess(self, preds):

        """Preprocesses the predictions."""

        return preds



    def init_metrics(self, model):

        """Initialize performance metrics for the YOLO model."""

        pass



    def update_metrics(self, preds, batch):

        """Updates metrics based on predictions and batch."""

        pass



    def finalize_metrics(self, *args, **kwargs):

        """Finalizes and returns all metrics."""

        pass



    def get_stats(self):

        """Returns statistics about the model's performance."""

        return {}



    def check_stats(self, stats):

        """Checks statistics."""

        pass



    def print_results(self):

        """Prints the results of the model's predictions."""

        pass



    def get_desc(self):

        """Get description of the YOLO model."""

        pass



    @property

    def metric_keys(self):

        """Returns the metric keys used in YOLO training/validation."""

        return []



    def on_plot(self, name, data=None):

        """Registers plots (e.g. to be consumed in callbacks)."""

        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}





    def plot_val_samples(self, batch, ni):

        """Plots validation samples during training."""

        pass



    def plot_predictions(self, batch, preds, ni):

        """Plots YOLO model predictions on batch images."""

        pass



    def pred_to_json(self, preds, batch):

        """Convert predictions to JSON format."""

        pass



    def eval_json(self, stats):

        """Evaluate and return JSON format of prediction statistics."""

        pass

