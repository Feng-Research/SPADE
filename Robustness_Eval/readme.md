SPADE_evaluation
===============================

SPADE: A Spectral Method for Black-Box Adversarial Robustness Evaluation

Usage
-----

**SPADE-Guided Robustness Evaluation**
1. To reproduce our experiment, extract data from following Models:

`mnist(modified from https://github.com/MadryLab/mnist_challenge)`

    1. config parameters and paths in config.json, mainly model_path and eps
    2. run train.py to train tensorflow model for mnist dataset(with adversial examples), the models are saved in checkpoints in ./models folder
    2b. model with eps=0 and 0.3 can also be downloaded using `fetch_model.py`, in this way: `python fetch_model.py natural`(or `adv_trained`)
    3a. when analyzing characteristic of model, run `gbt_eval.py` to evaluate with trained model through the train/test set(output paths can be set in the py file, while input and other config in config.json)
    3b. when runing for clever score, run model_converter.py to get the .h5 file containing structure and weights of the model. Paths are set at the head of the py file. 
        by default .5h model files are also stored in ./models folder

`robustness(modified from https://github.com/MadryLab/robustness)`

    1. install robustness package using pip or other similar tools
    2. download model from github page https://github.com/MadryLab/robustness, 
        or train model with robustness, e.g. 
            `python -m robustness.main --dataset restrict_imagenet  --adv-train 0 --arch resnet50 --out-dir logs/checkpoints/dir/`
        for a naturally trained resnet-50 model for cifar dataset, 
        more parameters see 
            https://robustness.readthedocs.io/en/latest/example_usage/cli_usage.html#training-a-standard-nonrobust-model
    3. evaluate through the model trained using `run.py`. 
        by default .pt model files are put together with `run.py`, and output files are stored in `./train_eval_results` foler


`CLEVER(modified from https://github.com/huanzhang12/CLEVER)`

    1. run `python3 collect_gradients.py --data mnist --model_name 2-layer --target_type 16 --numimg 10 -s ./your data storage location` to get gredients. Data set options: `mnist` and `cifar`. Model_name to "2-layer" (MLP), "normal" (7-layer CNN), "distilled" (7-layer CNN with defensive distillation). For `mnist`, two more Model_name are `mnist_0` and `mnist_03`
    1b(SPADE-Guided CLEVER training). to calculate SPADE-Guided gredients, an example is adding `--ids ./normal_spade_nodelist/vanilla/nodes_cifar_cnn.csv` to 1. you can switch all csv file to ids to calculate different network gredients. nodes rankings are calculated with 10NN, `vanilla` stands for top ranked nodes, `vanilla_reverse` stands for reversed top ranked nodes.
    2. run `python3 clever.py --untargeted ./your_data_storage_location/` to get clever score.
    
 `AugMix(modified from https://github.com/google-research/augmix)`
 
    1. run `python3 cifar.py -s ./your_data_storage_location/ -m networks_name --no-jsd ` for AugMix trained data
    2. run `python3 cifar.py -s ./your_data_storage_location/ -m networks_name --mix_off --no-jsd ` for standard trained data
       networks options have `allconv`, `resnext`, `densenet`, and `WRN`

`GAIRAT(modified from https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training.git)`
    
    1. run `python3 GAIRAT.py --epsilon 0.25 --net 'resnet18' --out-dir './your_data_save_location' `
