RF-DETR Small — P99.99/BG05 D4 model
======================================

This is the best RF-DETR checkpoint evaluated so far on the frozen held-out
P99.99/BG05 test split. It is the D4-only augmentation run; model selection
used validation mAP@50-95, not test performance.

Checkpoint
----------
weights/checkpoint_best_ema.pth

SHA-256
-------
325d040dd71820e3c7e82d79b6d6c4e2ef9833af7b360e825adda98b4925b4dc

Dataset and model
-----------------
- RF-DETR Small, RF-DETR 1.11.0
- Frozen P99.99/BG05 COCO dataset and split
- 640 px input; batch size 8; gradient accumulation 2
- D4 augmentation at p=0.75; scale jitter enabled
- Six classes, in order: prophase, earlyprometaphase, prometaphase,
  metaphase, anaphase, telophase

Checkpoint selection
--------------------
- Best validation epoch: 7 (one-based)
- Validation mAP@50-95: 0.6209859848022461
- Validation mAP@50: 0.752714216709137

Held-out test metrics
---------------------
- mAP@50-95: 0.6839404106140137
- mAP@50: 0.7860223054885864
- Precision: 0.7222534418106079
- Recall: 0.7500389814376831
- F1: 0.7302888631820679

Companion files
---------------
training_config.json records the full RF-DETR training configuration.
test_evaluation.json records the validation-only checkpoint-selection rule and
the held-out test metrics. The checkpoint is tracked with Git LFS.

Loading example
---------------
from rfdetr import RFDETRSmall
model = RFDETRSmall.from_checkpoint(
    "Models/rfdetr_small_p99p99_bg05_d4_v1/weights/checkpoint_best_ema.pth",
    trust_checkpoint=True,
)
