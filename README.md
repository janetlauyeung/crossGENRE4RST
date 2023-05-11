# EACL2023: Why Can’t Discourse Parsing Generalize? A Thorough Investigation of the Impact of Data Diversity
This README.md provides an overview of the code repositories contained here 
and instructions on running the experiments described in the [paper](https://aclanthology.org/2023.eacl-main.227/):  
```bash
@inproceedings{liu-zeldes-2023-cant,
    title = "Why Can{'}t Discourse Parsing Generalize? {A} Thorough Investigation of the Impact of Data Diversity",
    author = "Liu, Yang Janet  and
      Zeldes, Amir",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.227",
    pages = "3112--3130",
    abstract = "Recent advances in discourse parsing performance create the impression that, as in other NLP tasks, performance for high-resource languages such as English is finally becoming reliable. In this paper we demonstrate that this is not the case, and thoroughly investigate the impact of data diversity on RST parsing stability. We show that state-of-the-art architectures trained on the standard English newswire benchmark do not generalize well, even within the news domain. Using the two largest RST corpora of English with text from multiple genres, we quantify the impact of genre diversity in training data for achieving generalization to text types unseen during training. Our results show that a heterogeneous training regime is critical for stable and generalizable models, across parser architectures. We also provide error analyses of model outputs and out-of-domain performance. To our knowledge, this study is the first to fully evaluate cross-corpus RST parsing generalizability on complete trees, examine between-genre degradation within an RST corpus, and investigate the impact of genre diversity in training data composition.",
}

```

Overall, the directories included in this repository contains the adapted code 
for our experiments from the original implementation of 
[Guz and Carenini (2020)](https://www.aclweb.org/anthology/2020.codi-1.17/)'s 
base system, `SpanBERT-NoCoref`. 

Since our adaption to the original implementation contains several major changes 
to the original code, we suggest that you follow the general setup instructions compiled
below to run the experiments. We also added code (which has been integrated into the 
original code) to convert bracketing `.dis` output to `.rs3` files (`../src/utils/tree2rs3.py`) 
for visualization using [rstWeb](https://gucorpling.org/rstweb/info/).  




## TL;DR

1. The environment setup is needed for all experiments with no experiment-specific or dataset-specific requirements. 
Thus, follow the instructions below. 


2. `RST-DT` is licensed, and therefore you need to follow the preprocessing steps 
below **AFTER** obtaining a copy of the data. `GUM` is publicly available online 
[here](https://github.com/amir-zeldes/gum). You can preprocess the data on your own or obtain a copy of already-processed GUM data (GUM V8 used in the experiments presented in this paper) 
[here](https://drive.google.com/file/d/1XPBm03XA5QceYNtSf64EdBW9Z-AjGq7M/view?usp=share_link). 


3. We provide trained RST models (**We select models  with scores closest to average run scores reported in the paper**) and fine-tuned SpanBERT-base model as well as automatic GUM parses in 
`.rs3` and `.rsd`, which can be obtained from the following links respectively:
      - `GUM_parses`: [here](https://drive.google.com/drive/folders/1Wi9RZOoIaXF4If6sNfWS40fN_xO1y1_B?usp=share_link)
      - `The fine-tuned SpanBERT model used in our experiments`: [here](https://drive.google.com/file/d/1W4hTj0COS8VYliJLukCtWPKrYZ4zTwY8/view?usp=share_link)
        - Please place the entire downloaded folder (after unzipping it) under `../data/`, i.e. `../data/finetuned-spanbert`. 
      - `Models`: [here](https://drive.google.com/drive/folders/1e8bxGrWJbIWNPbmYcwWx9AaFaejSkf_P?usp=share_link)
        - If you'd like to use the trained model directly, 
        please place the content of the downloaded `.zip` file under `../data/`. 
        The overall structure should look like below 
        (the name of the trained model (`.pt`) and the data helper (`.bin`) varies across experiments):
        ```
        data/model/gum_train_model.pt
        data/gum_train_data_helper_rst.bin
        ```


   

## Directories


- ``GUM_experiments`` contains code for each GUM-related experiment described in Section 3, specifically: 
   ```bash
      - Section 3.1: Cross-Corpus Generalization (GUM) 
      - Section 3.3: OOD Multi-Genre Degradation
      - Section 3.4: Genre Variety in a Fixed-Size Sample
   ```


- ``GUM_parses`` contains a README.md that provides information about accessing automatic parses we obtained 
from GUM-related experiments.


- ``GUM_splits`` contains two `.txt` files that provide the established GUM V8 `train/dev/test` splits used in all 
our GUM-related experiments. Once obtaining a processed version of the GUM V8 data from the aforementioned link above, 
make the data folder accordingly (see more details below) based on the splits provided here. 


- ``RST-DT_experiments`` contains repositories for each RSTDT-related experiment in Section 3, specifically: 
   ```bash
      - rstdt_base: Section 3.1 (RSTDT) and Section 3.2 (CONCAT)
      - rstdt_label: Section 3.2 (SR-LABEL & FLAIR-LABEL)
      - rstdt_graph: Section 3.2 (SR-GRAPH)
      - rstdt_ft: Section 3.2 (SR-FT)
   ```




## General Setup Guidelines & Data Preprocessing Instructions
*Below, we use the ``rstdt_base`` directory as an example, which contains code used for the cross-corpus generalization 
on RST-DT in Section 3.1 and the `CONCAT` experiment in Section 3.2. 
For other experiments, simply change the name of target repository and update
the data composition / content accordingly. 
In the case of the GUM-related experiments, don't forget to change the 
root directory (i.e.``GUM_experiments``) as well.* 

1. Clone this repository with a Python environment of 3.6 or 3.8: 
   ```bash
   conda create --name ENV_NAME python=3.6
   conda activate ENV_NAME
   ```

2. Install dependencies
   ```bash
    cd rstdt_base/src/ubc_coref
    python -m pip install -e .
    cd rstdt_base/
    pip install -r requirements.txt
    ```
3. Obtain a copy of [the RST-DT data from LDC](https://catalog.ldc.upenn.edu/LDC2002T07) and place the **`data`** directory 
in parallel to the **`src`** directory
   ```bash
   data/train_dir/*
   data/test_dir/*
   src/
   ```
4. Stanford CoreNLP toolkit is used to preprocess the data, 
as part of the original implementation. Download it from 
[here](http://stanfordnlp.github.io/CoreNLP/index.html) 
and put the file [run_corenlp.sh](rstdt_base/run_corenlp.sh) into the CoreNLP folder. 
Then use the following command to preprocess both the data in the 
```train_dir``` and ```test_dir``` directories:
    ```bash
    python preprocess.py --data_dir DATA_DIR --corenlp_dir CORENLP_DIR
     ```
   Then, move the following filenames' processed files to a separate 
   repository ``data``, ```dev_dir```.
    ```bash
     ['wsj_0618', 'wsj_0621', 'wsj_0622', 'wsj_0634', 'wsj_0672',
     'wsj_0683', 'wsj_1104', 'wsj_1115', 'wsj_1118', 'wsj_1131',
     'wsj_1147', 'wsj_1154', 'wsj_1166', 'wsj_1167', 'wsj_1172',
     'wsj_1181', 'wsj_1193', 'wsj_1309', 'wsj_1310', 'wsj_1323',
     'wsj_1332', 'wsj_1349', 'wsj_1360', 'wsj_1371', 'wsj_1374',
     'wsj_1377', 'wsj_1397', 'wsj_1399', 'wsj_1963', 'wsj_2308',
     'wsj_2340', 'wsj_2350', 'wsj_2352', 'wsj_2364', 'wsj_2391']
    ```
    The ``data`` folder should then look like this: 
    ```
       data/dev_dir/*
       data/test_dir/*
       data/train_dir/*
       src/
    ```

5. Generally speaking, there are three data formats that is needed for each document and thus 
they must be present in the respective data directories:
   ````
   .merge
   .edus
   .dis
   ````




## Experiments

*Below we provide general steps to run an experiment: PREPARE, TRIAN, and TEST. 
Please make sure the right corresponding training / validation / test data directories
are provided. The following instructions exemplify training and testing on 
RST-DT. If you would like to evalaute this model on GUM ``test`` data, 
then make sure to create a new folder that contains GUM `test` files.*

1. Change to the right working directory 
   ```bash
   cd rstdt_base/src
   ```
   
2. **PREPARE**: Run the following to generate the action/relation maps
   ```bash
   python main.py 
   --prepare 
   --train_dir "../data/train_dir/"
   ```
   
3. **TRAIN**: 0 for the baseline model (no coreference)
   ```bash
   python main.py
   --train
   --model_name "train_dir_model.pt"
   --model_type 0
   --train_dir "../data/train_dir/"
   ```
   
4. **TEST**: The evaluation metric has been defaulted to the standard Parseval 
instead of RST-Parseval, so no specific parameter is needed, 
unlike the original implementation. 
   ```bash
   python main.py
   --eval
   --train_dir "../data/train_dir/"
   --eval_dir "../data/test_dir/"
   --model_name "train_dir_model.pt"
   --model_type 0
   --model_type 0
   ```
   OR
   ```bash
   python main.py
   --eval
   --train_dir "../data/train_dir/"
   --eval_dir "../data/gum_test/"
   --model_name "train_dir_model.pt"
   --model_type 0
   ```




## Notes
1. In order to reproduce our experiments, please remember to use GUM's established ``dev`` partition instead of 
randomly selecting documents from the ``train`` partition. 
The established splits of GUM V8 are provided in ``GUM_splits``. 

2. The conversion code we use to convert `.rs3` to `.rsd` can be found [here](https://github.com/amir-zeldes/rst2dep). 
