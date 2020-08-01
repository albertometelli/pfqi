# Control Frequency Adaptation via Action Persistence in Batch Reinforcement Learning

This repository contains the code for our [paper](https://arxiv.org/abs/2002.06836) Control Frequency Adaptation via Action Persistence in Batch Reinforcement Learning, which was accepted at ICML 2020. The FQI implementation is derived from [TRLIB](https://github.com/AndreaTirinzoni/iw-transfer-rl).

### Requirements
```
Python 3
numpy
scikit-learn
joblib
matplotlib
gym
```

### Example
To run PFQI for different persistences in the Cartpole environment:
```
python3 scripts/run_cartpole.py
```
The results will be stored in a json file. To plot the results:
```
python3 plotters/multi_perf_plotter4x4.py plotters/example.json
```

### Citing
```
@incollection{metelli2020control,
    author = "Metelli, Alberto Maria and Mazzolini, Flavio and Bisi, Lorenzo and Sabbioni, Luca and Restelli, Marcello",
    booktitle = "Proceedings of the 37th International Conference on Machine Learning, Online, PMLR 119, 2020",
    pages = "4102--4113",
    title = "Control Frequency Adaptation via Action Persistence in Batch Reinforcement Learning",
    year = "2020",
} 
```