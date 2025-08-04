# README

Our implementation is based on OpenPCDet(https://github.com/open-mmlab/OpenPCDet/tree/master).

For the evironment setup, please refer to  https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md

To train the model, use the command
`python train.py --cfg_file cfgs/nuscenes_models/counter_net.yaml --epochs=10`

For the training of different partition and overlap, please replace the yaml file under `cfgs/nuscenes_models/`

The training scripts for KITTI and Waymo, please find the yaml file under `cfgs/kitti_models` and `cfg/waymo_models`, respectively.

To evaluate the result, please run
`bash scripts/dist_test.sh 1 --cfg_file cfgs/nuscenes_models/counter_net.yaml --ckpt  /path_to_model`
After that, please run eval_counter.py.
