python main_cls.py \
--root_dir data/ClipShots/Videos \
--image_list_path data/data_list/deepSBD.txt \
--result_dir results \
--model resnet \
--n_classes 3 --batch_size 32 --n_threads 1 \
--sample_duration 16 \
--learning_rate 1e-3 \
--gpu_num 1 \
--manual_seed 16 \
--shuffle \
--spatial_size 128 \
--pretrain_path kinetics_pretrained_model/resnet-18-kinetics.pth \
--gt_dir data/ClipShots/Annotations/test.json \
--test_list_path data/ClipShots/Video_lists/test.txt \
--total_iter 3000 \
--auto_resume |tee results/test.log