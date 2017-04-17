# morcegosExperiments

### Pre-requisitos ###

* Tensorflow [1]
* scikit-learn [2]
* scipy [3]
* pandas [4]
* numpy
* python
* opencv [5]
* pyAudioAnalysis [6]

### Funcionamento ###

O arquivo main.py executa os experimentos usando o algoritmo DFA e usando as features (34) extraídas pelo extrator do pyAudioAnalysis.

O arquivo networkTensorflow.py executa os experimentos usando o modelo de Deep Learning, tendo como entrada imagens 56x92 em grayscale. Em reader.py estão funções de leitura e criação do dataset para os experimentos

## Links úteis
```
* [1] - https://www.tensorflow.org
* [2] - http://scikit-learn.org/stable/
* [3] - https://www.scipy.org
* [4] - http://pandas.pydata.org
* [5] - http://opencv.org
* [6] - https://github.com/tyiannak/pyAudioAnalysis/
```