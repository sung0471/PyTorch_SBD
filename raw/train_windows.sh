# for cmd
python main_cls.py ^
--root_dir data/ClipShots/Videos ^
--image_list_path data/data_list/deepSBD.txt ^
--result_dir results ^
--model resnet ^
--n_classes 3 --batch_size 60 --n_threads 16 ^
--sample_duration 16 ^
--learning_rate 0.001 ^
--gpu_num 1 ^
--manual_seed 16 ^
--shuffle ^
--spatial_size 128 ^
--pretrain_path kinetics_pretrained_model/resnet-18-kinetics.pth ^
--gt_dir data/ClipShots/Annotations/test.json ^
--test_list_path data/ClipShots/Video_lists/test.txt ^
--total_iter 300000 ^
--auto_resume | powershell "tee results/test.log"

# for powershell
python main_cls.py `
--root_dir data/ClipShots/Videos `
--image_list_path data/data_list/deepSBD.txt `
--result_dir results `
--model resnet `
--n_classes 3 --batch_size 60 --n_threads 16 `
--sample_duration 16 `
--learning_rate 0.001 `
--gpu_num 1 `
--manual_seed 16 `
--shuffle `
--spatial_size 128 `
--pretrain_path kinetics_pretrained_model/resnet-18-kinetics.pth `
--gt_dir data/ClipShots/Annotations/test.json `
--test_list_path data/ClipShots/Video_lists/test.txt `
--total_iter 300000 `
--auto_resume |tee results/test.log