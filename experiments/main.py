import extractor as e
import unet as unet
import unet800k as unet2
import unet200k as unet3
import unet80k as unet4
import unet14k as unet5
from analysis import run_mask_quality_analysis
from sam_comparison import run_sam_comparison
from sam_training import generate_sam_tp_dataset, retrain_previous_models_on_tp
from inference_time import run_model_benchmark
if __name__ == '__main__':
    # e.create_1k_subset()
    # run_mask_quality_analysis()
    run_model_benchmark()
    # unet.run_unet_experiments()
    # unet2.run_efficient_unet_experiments()
    # unet3.run_micro_cow_unet_experiments()
    # unet4.run_nano_cow_unet_experiments()
    # unet5.run_pico_cow_unet_experiments()
    # run_sam_comparison()
    # generate_sam_tp_dataset(iou_threshold=0.25)

# ==========================================================================================
# SAM2 vs COCO COMPARISON FINISHED!
# Matched 1000 images out of 1000 in the 1K subset
# Results saved → sam_comparison\sam_metrics.csv
#
# Average metrics (± std):
#   IoU       : 0.1503 ± 0.1092
#   Dice      : 0.2466 ± 0.1558
#   Precision : 0.1556 ± 0.1082
#   Recall    : 0.7153 ± 0.3686
#   F1-score  : 0.2466 ± 0.1558
# ==========================================================================================
