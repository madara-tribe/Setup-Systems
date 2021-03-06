# sample_pakage_model

This is code to sample pakaged model of classify images.

<b>Folder sturacture</b>
```
$ tree

├── README.md
├── requirements.txt
├── setup.py
├── setup.sh
└── trainer
    ├── __init__.py
    ├── models
    │   ├── __init__.py
    │   ├── cnn_model.py
    │   └── scse.py
    └── train.py
```



# Way to package in virtual environment

```
# create virtual environment(ENV=sample)
./setup.sh

# create virtual environment
cp -r trainer setup.py requirements.txt ${ENV}/ && cd ${ENV}

# install modules
pip3 install -r requirements.txt

# check whether modules in requirements.txt were installed 
pip3 freeze  

# make package to provide
python3 setup.py sdist
```


# memo
「module」is classses and functions in scripts
「pakkage」is pakkaging modules
in this codes, pakkage is 'trainer'.
フォルダ内にscriptがあることをindicateするために「__init__.py」をフォルダ内に作る。
for example to import in script
```
from trainer import train 
from trainer.models import cnn_models
import trainer.train
```
# reference

[how to make module1](https://uxmilk.jp/41603)

[how to make module2](https://qiita.com/Kensuke-Mitsuzawa/items/7717f823df5a30c27077)

[exsample-repository](https://github.com/fizyr/keras-maskrcnn)
