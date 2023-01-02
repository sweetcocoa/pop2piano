# Pop2Piano : Pop Audio-based Piano Cover Generation
---
##
- [Paper](https://arxiv.org/abs/2211.00895)
- [Colab](http://bit.ly/pop2piano-colab)
- [Project Page](http://sweetcocoa.github.io/pop2piano_samples)

## Run locally from source

Install dependencies (Debian and Ubuntu based distributions):
`sudo apt install fluidsynth`

Arch based distributions:
`sudo apt install fluidsynth`

Download the code and model:
```
cd
mkdir -p src
cd src
git clone https://github.com/sweetcocoa/pop2piano/
cd pop2piano
wget https://github.com/sweetcocoa/pop2piano/releases/download/dpi_2k_epoch/model-1999-val_0.67311615.ckpt
```
It is not necessary to download to the `src` folder; if you do, the default file paths will work.
If you don't, you'll need to specify the path to the `yaml` and `model` files, 
either in your command or in the config file.

If you are running virtual environment, create one and activate it.
This has been tested on Python 3.8.
Then install python dependencies:
```
pip install -r requirements.txt
```

Then, run it:
```
python -m pop2piano.main -c 2,18 -o outputs /path/to/inputs
```
You can also run 
```
python -m pop2piano.main -h
```
to get a help message with all arguments.

## Install

Following the install guide, at the root of the project, run:
```
pip install build
```
To install build dependency, then build it:
```
python -m build
```
This creates a `dist` folder that you can install from:
```
pip install ./dist/pop2piano*.whl
``` 
(you can also replace `pip` by `pipx`.)

## How to prepare dataset
### Download Original Media
---
- List of data : ```train_dataset.csv```
- Downloader : ```download/download.py```
    - ```python download.py ../train_dataset.csv output_dir/```

### Preprocess Data
---
- [Details](./preprocess/)





