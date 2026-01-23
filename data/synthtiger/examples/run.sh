source /mnt/bn/ocr-doc-nas/zhuhanshen/home/anaconda3/etc/profile.d/conda.sh #replace with your own

# 激活你的 conda 环境，这里假设环境名为 tiger
conda activate tiger
cd /mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger #replace with your own

# usage: synthtiger [-h] [-o DIR] [-c NUM] [-w NUM] [-s NUM] [-v] SCRIPT NAME [CONFIG]

# positional arguments:
#   SCRIPT                Script file path.
#   NAME                  Template class name.
#   CONFIG                Config file path.

# optional arguments:
#   -h, --help            show this help message and exit
#   -o DIR, --output DIR  Directory path to save data.
#   -c NUM, --count NUM   Number of output data. [default: 100]
#   -w NUM, --worker NUM  Number of workers. If 0, It generates data in the main process. [default: 0]
#   -s NUM, --seed NUM    Random seed. [default: None]
#   -v, --verbose         Print error messages while generating data.

# Feel free to modify the commands and config files to meet your own requirements

#英文文本图像合成
tiger -o syth_multitext_en -w 16 -v examples/synthtiger/template_en_multiline.py SynthTigerEN examples/synthtiger/config_multiline_en.yaml  -c 50

#中文文本图像合成
tiger -o syth_multitext_p2 -w 16 -v examples/synthtiger/template_zn_multiline.py SynthTigerZH examples/synthtiger/config_multiline_zh.yaml  -c 50

#含结构错误的中文文本图像合成
tiger -o syth_multitext_zaozi -w 16 -v examples/synthtiger/template_zn_multiline.py SynthTigerZH examples/synthtiger/config_multiline_zh_zaozi.yaml  -c 50



