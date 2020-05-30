# Shot Boundary Detection using 3D CNN
- This repository contains our implementation of Shot Boundary Detection.
- The code is modified from here
    - Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?
	[paper](https://arxiv.org/abs/1711.09577),
	[code](https://github.com/kenshohara/3D-ResNets-PyTorch)
    - Fast Video Shot Transition Localization with Deep Structured Models
	[paper](https://arxiv.org/pdf/1808.04234.pdf),
	[code](https://github.com/Tangshitao/ClipShots_basline)

## Requirement
    requirement.txt
    
    or
    
    Window 10
    conda create--name pytorch100 python=3.6
    conda install pytorch=1.0.1 torchvision=0.2.2 cudatoolkit=10.0 -c pytorch
    conda install -c menpo ffmpeg=2.7.0
    conda install -c conda-forge opencv=3.4.2
    conda install tensorboardx=1.6
    conda install scikit-learn=0.20.3
    pip install matplotlib==3.0.3
    pip install tensorflow==1.15.2
    pip install thop==0.0.23

## Dataset
- directory structure
    
        data/[dataset_name]/
            annotations/    : annotations of videos
            video_lists/    : video file name list
            videos/         : video files
            
1. ClipShots Dataset ([From here](https://github.com/Tangshitao/ClipShots))
    - directory
    
    		data/ClipShots/
        
    - clone above repository to directory `data/`
    - download videos from above repository
2. RAI Dataset ([From here](http://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=019))
    - directory
    
    		data/RAI/
        
    - save videos to `data/RAI/videos/test`
    - prepare annotations and video_lists (not now in repository)
3. TRECVID Dataset
    - directory
    
    		data/TRECVID/[Year]/
    
    - In my case, we use TRECVID 2007 dataset. following description is about TRECVID 2007
    1. request TRECVID videos through following [form](https://www-nlpir.nist.gov/projects/trecvid/SV.2010.forms/)
        - download videos in shot.test provided from NIST
        - save videos to `data/TRECVID/07/videos/test`
    2. if you don't have annotations,
        - download annotations from [here](https://trecvid.nist.gov/trecvid.data.html#tv07)
        - unzip `sbref07.1.tar.gz` and save files to `data/TRECVID/07/`
        - and then, execute `make_trecvid_dataset.py`
            - copy TRECVID videos from other directory to `data/TRECVID/07/videos/test` when no video in `test` folder
            - make annotations JSON file using XML files from `/ref/`

## Resources
1. pre-trained Shot Boundary Detection models
    - used models in this repository ([GitHub](https://github.com/Tangshitao/ClipShots_basline))
        1. The trained model for Alexnet-like backbone. [BaiduYun](https://pan.baidu.com/s/16q3CNuUhLAGkm21PPOqUSg), [Google Drive](https://drive.google.com/open?id=145NCxLhgdrKPIYm-qgp1SRYU_GFmzxxX)
        2. The trained model for ResNet-18 backbone. [BaiduYun](https://pan.baidu.com/s/1Bx2uVVQOuEnTxdBBGV3uCQ), [Google Drive](https://drive.google.com/file/d/1CVqxAp17OOBmNq9_jgEdaoDbrmK5Bmog/view?usp=sharing)
    - save to `models/`
2. pre-trained kinetics models
    - You can download pre-trained models ([Google Drive](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M))
    - used models in this repository
        1. resnet-18-kinetics.pth
        2. resnet-50-kinetics.pth
        3. resnext-101-kinetics.pth
    - save to directory `kinetics_pretrained_model/`

## Experiment result

### Dataset
use ClipShots and TRECVID 2007 Dataset. no RAI

### Model([Google Drive](https://drive.google.com/drive/folders/1iWtSmHIagl5SourwSM_BOyYJqiKbeKH9?usp=sharing))
1. resnext-101
	1. no pre-trained kinetics model([Google Drive](https://drive.google.com/file/d/1cXk9eYTb9BMcTSHucN0xUqrD6b0suVoI/view?usp=sharing))
		- command (example)

	          python main_baseline.py --phase test --dataset ClipShots --test_weight results/model_final/model_final_resnext_noPre_epoch5.pth --train_data_type normal --model resnext --model_depth 101 --pretrained_model False --loss_type normal --sample_size 128

	2. use pre-trained kinetics model([Google Drive](https://drive.google.com/file/d/1xAMws_Tw4gMiSzcwD2c3lRiz9Fx_rN9G/view?usp=sharing))
		- command (example)

              python main_baseline.py --phase test --dataset ClipShots --test_weight results/model_final/model_final_resnext_epoch5.pth --train_data_type normal --model resnext --model_depth 101 --pretrained_model True --loss_type normal --sample_size 128

2. Knowledge distillation(teacher: alexnet, student: resnext-101)
	1. no pre-trained kinetics model([Google Drive](https://drive.google.com/file/d/19eZ7GZ7LaZeBcRpHP_vtDAxdun5oee7_/view?usp=sharing))
		- command (example)

              python main_baseline.py --phase test --dataset ClipShots --test_weight results/model_final/model_final_teacher_noPre_epoch5.pth --train_data_type normal --model resnext --model_depth 101 --pretrained_model False --loss_type KDloss --sample_size 128

	2. use pre-trained kinetics model([Google Drive](https://drive.google.com/file/d/1j5vCuZTxH-krPQG1_qU6K6Q12_gUYIQ-/view?usp=sharing))
		- command (example)

		      python main_baseline.py --phase test --dataset ClipShots --test_weight results/model_final/model_final_teacher_epoch5.pth --train_data_type normal --model resnext --model_depth 101 --pretrained_model True --loss_type KDloss --sample_size 128

3. detector (use pre-trained kinetics model)([Google Drive](https://drive.google.com/file/d/1QtsQduLUDsz4aE5cDJRJxkMRHwmuUpgb/view?usp=sharing))
	- command (example)

	      python main_baseline.py --phase test --dataset ClipShots --test_weight results/model_final/model_final_detector_epoch5.pth --train_data_type cut --layer_policy second --model detector --baseline_model resnet --model_depth 50 --pretrained_model True --loss_type multiloss --sample_size 64


## Traning and Testing
#### - Change options in files <code>opts.py</code> and <code>utils/config.py</code> or
#### - run command python

1. Training
    - options
        - phase

              --phase train

        - Dataset (only ClipShots)

              --dataset ClipShots

        - Model

              - alexnet / resnet / resnext
            
                  --model alexnet --train_data_type normal --loss_type normal
                  --model resnet -model_depth 18 --train_data_type normal --loss_type normal
                  --model resnext --model_depth 101 --train_data_type normal --loss_type normal

              - detector
                
                  --model detector --baseline_model resnet --model_depth 50 --train_data_type cut --layer_policy second --loss_type multiloss --sample_size 64

        - use/no pre-trained model

              --pretrained_model True/False

        - loss function

              - normal (only classification)
                  --loss_type normal

              - KDloss (only classification)
                  --loss_type KDloss
              
              - multiloss (classification + regression)
                  --loss_type multiloss

    - command (example)

          python main_baseline.py --phase train --dataset ClipShots  --model detector --baseline_model resnet --model_depth 50 --pretrained_model True --loss_type multiloss --sample_size 64

2. Testing : `opts.py` → phase='test'
    - options
        - phase
            
              --phase test
        
        - Dataset
        
              --dataset ClipShots
              --dataset RAI
              --dataset TRECVID[year]
                ex) TRECVID07 (only 2007 year possible)
        
        - Model

              - alexnet / resnet / resnext
            
                  --model alexnet --train_data_type normal --loss_type normal
                  --model resnet -model_depth 18 --train_data_type normal --loss_type normal
                  --model resnext --model_depth 101 --train_data_type normal --loss_type normal

              - detector
                
                  --model detector --baseline_model resnet --model_depth 50 --train_data_type cut --layer_policy second --loss_type multiloss --sample_size 64

        - use/no pre-trained model
        
              --pretrained_model True/False

    - command (example)

          python main_baseline.py --phase test --dataset ClipShots --test_weight results/model_final/model_final_detector_epoch5.pth --train_data_type cut --layer_policy second --model detector --baseline_model resnet --model_depth 50 --pretrained_model True --loss_type multiloss --sample_size 64

3. Training and Testing
    
    - phase='full'

4. Running code
 
    - if change files, Run `run_[OS_type].sh`
    - else, Run python file `main_baseline.py` with options
