"""Two frozen variants. All initial weights originate from COCO yolo11n.pt."""
import copy

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from structure_feasibility import (FoldableContrastDW, add_contrast,
                                   align_classifier_width, lighten_p2_regression,
                                   p2_cfg)


class StructureDetectionModel(DetectionModel):
    def fuse(self, verbose=True, imgsz=640):
        for m in list(self.modules()):
            if isinstance(m, FoldableContrastDW) and not hasattr(m, 'reparam'):
                m.switch_to_deploy()
        return super().fuse(verbose=verbose, imgsz=imgsz)


def transform_base(base, variant):
    """Input is the standard nc=1 COCO-loaded baseline, never a CST checkpoint."""
    assert variant in ['A_zero_deploy', 'B_p2_compact']
    records = []
    if variant == 'A_zero_deploy':
        model = base
        model.__class__ = StructureDetectionModel
        records.append(dict(source='standard_COCO_loaded_baseline', destination='all original tensors',
                            tensors=len(base.state_dict())))
    else:
        model = StructureDetectionModel(p2_cfg(base.yaml, 32), nc=1, ch=3, verbose=False)
        model.names = copy.deepcopy(base.names)
        width = base.model[-1].cv3[0][0][1].conv.out_channels
        assert width == 80, width
        align_classifier_width(model, width)
        lighten_p2_regression(model, compact=True)
        initialize_weights(model)
        model.model[-1].bias_init()

        def transfer(dst, src, dst_name, src_name):
            d = dst.state_dict()
            s = src.state_dict()
            selected = {k: v for k, v in s.items() if k in d and v.shape == d[k].shape}
            dst.load_state_dict(selected, strict=False)
            records.append(dict(source=src_name, destination=dst_name,
                                copied_keys=sorted(selected),
                                copied_numel=sum(v.numel() for v in selected.values()),
                                destination_state_numel=sum(v.numel() for v in d.values())))
        # Backbone and original top-down neck retain their precise semantic roles.
        for i in range(17):
            transfer(model.model[i], base.model[i], f'model.{i}', f'model.{i}')
        # P3 refinement: reuse only compatible tensors; its new input width differs.
        transfer(model.model[22], base.model[16], 'model.22_P3_refinement', 'model.16_P3_refinement')
        transfer(model.model[25], base.model[19], 'model.25_P4_refinement', 'model.19_P4_refinement')
        dst_head, src_head = model.model[-1], base.model[-1]
        for new_level, old_level in [(1, 0), (2, 1)]:
            for tower in ['cv2', 'cv3']:
                transfer(getattr(dst_head, tower)[new_level], getattr(src_head, tower)[old_level],
                         f'head.{tower}.{new_level}', f'head.{tower}.{old_level}')
        transfer(dst_head.dfl, src_head.dfl, 'head.dfl', 'head.dfl')
    replacements = add_contrast(model)
    initialize_weights(model)
    model.structure_variant = variant
    model.structure_transfer = records
    model.structure_replacements = replacements
    model.yaml['structure_variant'] = variant
    for m in model.model:
        m.np = sum(p.numel() for p in m.parameters())
    return model


def build_from_source(source, variant, names=None):
    """Standalone preflight builder with identical name remapping to the trainer."""
    base = DetectionModel(copy.deepcopy(source.yaml), nc=1, ch=3, verbose=False)
    base.names = names or {0: 'uav'}
    base.load(source, verbose=False)
    return transform_base(base, variant)
