python test_cls.py \
--root_dir data/ClipShots/Videos \
--result_dir results \
--model alexnet \
--n_classes 3 --batch_size 32 \
--sample_duration 16 \
--spatial_size 128 \
--gt_dir data/ClipShots/Annotations/test.json \
--test_list_path data/ClipShots/Video_lists/test.txt \
--weights models/Alexnet-final.pth \
--auto_resume |tee results/test.log