



from pathlib import Path



import numpy as np

import torch



from ultralytics.models.yolo.detect import DetectionValidator

from ultralytics.utils import LOGGER, ops

from ultralytics.utils.metrics import OBBMetrics, batch_probiou

from ultralytics.utils.plotting import output_to_rotated_target, plot_images





class OBBValidator(DetectionValidator):

    """Validator for Oriented Bounding Box (OBB) models + APXS/APS/APM/APL (COCO-style ignore)."""



    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):



        super().__init__(dataloader=dataloader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks)



        self.args.task = "obb"



        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot, names=self.names)





        self._gt_offset = 0

        self._stats_matches = []

        self._stats_target_area = []



    def init_metrics(self, model):

        super().init_metrics(model)

        val = self.data.get(self.args.split, "")

        self.is_dota = isinstance(val, str) and "DOTA" in val



        self._gt_offset = 0

        self._stats_matches.clear()

        self._stats_target_area.clear()









    def get_desc(self):

        return ("%22s" + "%11s" * 10) % (

            "Class",

            "Images",

            "Instances",

            "Box(P",

            "Box(R",

            "Box(mAP50",

            "Box(mAP50-95",

            "APXS",

            "APS",

            "APM",

            "APL",

        )









    def _build_matches(self, pred_cls: torch.Tensor, gt_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:

        """
        matches[p, j] = matched GT index (local, 0..M-1) at IoU threshold j, else -1
        pred_cls: (N,)
        gt_cls:   (M,)
        iou:      (M,N)
        """

        n_pred = int(pred_cls.shape[0])

        niou = int(self.iouv.numel())

        matches = torch.full((n_pred, niou), -1, dtype=torch.long, device=iou.device)

        if n_pred == 0 or int(gt_cls.shape[0]) == 0:

            return matches



        cls_ok = (gt_cls[:, None] == pred_cls[None, :])



        for j, thr in enumerate(self.iouv):

            mask = (iou >= thr) & cls_ok

            gi, pj = torch.where(mask)

            if gi.numel() == 0:

                continue



            ious = iou[gi, pj]

            order = torch.argsort(ious, descending=True)

            gi = gi[order]

            pj = pj[order]



            used_g = set()

            used_p = set()

            for g, p in zip(gi.tolist(), pj.tolist()):

                if (g in used_g) or (p in used_p):

                    continue

                used_g.add(g)

                used_p.add(p)

                matches[p, j] = g



        return matches



    def _process_batch(self, det, gt_bboxes, gt_cls):

        """
        det:      (N,7) -> x,y,w,h,conf,cls,angle   (same coord system as gt_bboxes)
        gt_bboxes:(M,5) -> x,y,w,h,angle
        """



        iou = batch_probiou(gt_bboxes, torch.cat([det[:, :4], det[:, -1:]], dim=-1))





        correct = self.match_predictions(det[:, 5], gt_cls, iou)





        matches = self._build_matches(det[:, 5].to(gt_cls.dtype), gt_cls, iou)

        return correct, matches









    def _prepare_batch(self, si, batch):

        """GT: normalized -> imgsz pixel -> scale back to ori_shape pixel using ratio_pad."""

        idx = batch["batch_idx"] == si

        cls = batch["cls"][idx].squeeze(-1)

        bbox = batch["bboxes"][idx].clone()

        ori_shape = batch["ori_shape"][si]

        imgsz = batch["img"].shape[2:]

        ratio_pad = batch["ratio_pad"][si]



        if cls.shape[0]:

            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])



            ops.scale_boxes(imgsz, bbox[:, :4], ori_shape, ratio_pad=ratio_pad, xywh=True)



        return {

            "cls": cls,

            "bboxes": bbox,

            "ori_shape": ori_shape,

            "imgsz": imgsz,

            "ratio_pad": ratio_pad,

            "im_file": batch["im_file"][si],

        }









    def _prepare_pred_safe(self, pred, pbatch):

        """
        兼容不同 ultralytics 版本签名：
          - 有的 _prepare_pred(self, pred, pbatch)
          - 有的 _prepare_pred(self, pred)
        返回统一的 Tensor (N,7): x,y,w,h,conf,cls,angle
        """

        try:

            predn = super()._prepare_pred(pred, pbatch)

        except TypeError:

            predn = super()._prepare_pred(pred)





        if isinstance(predn, dict):



            b = predn.get("bboxes", None)

            a = predn.get("extra", None)

            c = predn.get("conf", None)

            k = predn.get("cls", None)

            if b is None or a is None or c is None or k is None:

                return torch.zeros((0, 7), device=self.device)

            predn = torch.cat([b, c.unsqueeze(1), k.unsqueeze(1), a], dim=1)





        if not isinstance(predn, torch.Tensor) or predn.ndim != 2 or predn.shape[1] < 7:

            return torch.zeros((0, 7), device=self.device)

        return predn[:, :7]









    def plot_predictions(self, batch, preds, ni):

        plot_images(

            batch["img"],

            *output_to_rotated_target(preds, max_det=self.args.max_det),

            paths=batch["im_file"],

            fname=self.save_dir / f"val_batch{ni}_pred.jpg",

            names=self.names,

            on_plot=self.on_plot,

        )



    def pred_to_json(self, predn, filename):

        stem = Path(filename).stem

        image_id = int(stem) if stem.isnumeric() else stem

        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)

        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)

        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):

            self.jdict.append(

                {

                    "image_id": image_id,

                    "category_id": self.class_map[int(predn[i, 5].item())],

                    "score": round(predn[i, 4].item(), 5),

                    "rbox": [round(x, 3) for x in r],

                    "poly": [round(x, 3) for x in b],

                }

            )



    def save_one_txt(self, predn, save_conf, shape, file):

        import numpy as np

        from ultralytics.engine.results import Results



        rboxes = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)

        obb = torch.cat([rboxes, predn[:, 4:6]], dim=-1)

        Results(

            np.zeros((shape[0], shape[1]), dtype=np.uint8),

            path=None,

            names=self.names,

            obb=obb,

        ).save_txt(file, save_conf=save_conf)



    def eval_json(self, stats):

        return stats









    @torch.no_grad()

    def update_metrics(self, preds, batch):

        for si, pred in enumerate(preds):

            self.seen += 1



            pbatch = self._prepare_batch(si, batch)

            tcls = pbatch["cls"]

            gt_bboxes = pbatch["bboxes"]





            predn = self._prepare_pred_safe(pred, pbatch)



            nl = int(tcls.shape[0])

            npr = int(predn.shape[0])



            if nl and npr:

                correct, matches_local = self._process_batch(predn, gt_bboxes, tcls)

            else:

                correct = torch.zeros((npr, self.niou), dtype=torch.bool, device=self.device)

                matches_local = torch.full((npr, self.niou), -1, dtype=torch.long, device=self.device)





            if npr:

                self.stats["tp"].append(correct.cpu())

                self.stats["conf"].append(predn[:, 4].cpu())

                self.stats["pred_cls"].append(predn[:, 5].cpu())

            else:

                self.stats["tp"].append(correct.cpu())

                self.stats["conf"].append(torch.zeros(0))

                self.stats["pred_cls"].append(torch.zeros(0))



            self.stats["target_cls"].append(tcls.cpu())

            self.stats["target_img"].append(

                tcls.unique().cpu().to(torch.int64) if nl else torch.zeros(0, dtype=torch.int64)

            )





            matches_global = matches_local.clone()

            if nl and npr:

                ok = matches_global >= 0

                matches_global[ok] += self._gt_offset





            if nl:

                target_area = (gt_bboxes[:, 2] * gt_bboxes[:, 3]).detach().cpu().numpy().astype(float)

            else:

                target_area = np.zeros((0,), dtype=float)



            self._stats_matches.append(matches_global.detach().cpu().numpy().astype(np.int64, copy=False))

            self._stats_target_area.append(target_area)



            self._gt_offset += nl



    def print_results(self):

        """
        Print one 'all' row with APXS/APS/APM/APL appended in the same line.
        (Do NOT call super().print_results(), otherwise it prints a row first.)
        """

        try:



            tp = torch.cat(self.stats["tp"], 0).cpu().numpy() if len(self.stats["tp"]) else np.zeros((0, self.niou),

                                                                                                     bool)

            conf = torch.cat(self.stats["conf"], 0).cpu().numpy() if len(self.stats["conf"]) else np.zeros((0,), float)

            pred_cls = (

                torch.cat(self.stats["pred_cls"], 0).cpu().numpy() if len(self.stats["pred_cls"]) else np.zeros((0,),

                                                                                                                float)

            )

            target_cls = (

                torch.cat(self.stats["target_cls"], 0).cpu().numpy() if len(self.stats["target_cls"]) else np.zeros(

                    (0,), float)

            )



            matches = (

                np.concatenate(self._stats_matches, 0) if len(self._stats_matches) else np.zeros((0, self.niou),

                                                                                                 np.int64)

            )

            target_area = (

                np.concatenate(self._stats_target_area, 0) if len(self._stats_target_area) else np.zeros((0,), float)

            )





            self.metrics.process(tp, conf, pred_cls, target_cls, matches=matches, target_area=target_area)





            p, r, map50, map = self.metrics.mean_results()



            apxs, aps, apm, apl = self.metrics.apxs, self.metrics.aps, self.metrics.apm, self.metrics.apl



            images = int(self.seen)

            instances = int(target_cls.shape[0])





            LOGGER.info(self.get_desc())

            LOGGER.info(

                ("%22s" + "%11i" * 2 + "%11.4f" * 8)

                % ("all", images, instances, p, r, map50, map, apxs, aps, apm, apl)

            )



        except Exception as e:

            LOGGER.warning(f"[APXS/APS/APM/APL] compute failed: {e}")



