SPADE
===============================

SPADE is a spectral method for black-box adversarial robustness evaluation. We propose model SPADE score, which is proved to be an upper bound of the best Lipschitz constant under the manifold setting, to capture non-robustness of ML models. Moreover, we further introduce node SPADE score to measure non-robustness of input samples, which is then used to guide applications such as adversarial training and robustness evaluation. More details are available in our paper: http://proceedings.mlr.press/v139/cheng21a.html

![Overview of the SPADE](/SPADE.png)

Citation
------------
If you use SPADE in your research, please cite our preliminary work
published in ICML'21.

```
@inproceedings{cheng2021spade,
title={SPADE: A Spectral Method for Black-Box Adversarial Robustness Evaluation},
author={Wuxinlin Cheng and Chenhui Deng and Zhiqiang Zhao and Yaohui Cai and Zhiru Zhang and Zhuo Feng},
booktitle={International Conference on Machine Learning},
year={2021},
url={http://proceedings.mlr.press/v139/cheng21a.html}
}
```

Usage
-----
**SPADE-Score Calculation Usage**
`cd SPADE_score/`

**SPADE-Guided Adversarial Training Usage**

`cd adv_train/`

**SPADE-Guided Robustness Evaluation Usage**

`cd Robustness_Eval/`
