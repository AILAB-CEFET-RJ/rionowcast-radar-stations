"""python main.py \
  -m stconvs2s-c \
  -dsp /home/noemi/atmoseer/data/datasets/dataset_radar_ws_masked_15min_256_5in_5out_stride5.npz \
  -b 2 \
  -e 10 \
  -i 1 \
  -w 0 \
  -s 5 \
  --verbose
"""

import subprocess

DATASET_PATH = (
    "/home/noemi/atmoseer/data/datasets/"
    "dataset_radar_ws_masked_15min_256_5in_5out_stride5.npz"
)

command = [
    "python",
    "/home/noemi/atmoseer/stconvs2s/main.py",
    "-m", "stconvs2s-c",
    "-dsp", DATASET_PATH,
    "-b", "2",
    "-e", "10",
    "-i", "1",
    "-w", "0",
    "-s", "5",
    "--verbose"
]

subprocess.run(command, check=True)