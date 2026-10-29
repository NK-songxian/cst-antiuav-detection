"""Trainer plug-in; ordinary YOLO detection criterion and dataset transforms."""
from ultralytics.models.yolo.detect import DetectionTrainer
from structure_model import transform_base


class StructureTrainer(DetectionTrainer):
    variant = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        assert weights is not None, 'COCO initialization required'
        assert weights.model[-1].nc == 80, 'Use original COCO weights; no CST warm start'
        base = super().get_model(cfg=weights.yaml, weights=weights, verbose=False)
        return transform_base(base, self.variant)


class TaskATrainer(StructureTrainer):
    variant = 'A_zero_deploy'


class TaskBTrainer(StructureTrainer):
    variant = 'B_p2_compact'
