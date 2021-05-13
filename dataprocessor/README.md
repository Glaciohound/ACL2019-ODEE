# ODEE Data Preprocessor
An tool to preprocess GN dataset into Staford CoreNLP XML output format 

```
export SCRIPTPATH=$(pwd)
cd ./dataprocessor
```

python3 for data preprocessing

run `pip3 install -r requirements.txt` to install requirements.

## Preprocess GN dataset

### Prepare Stanford CoreNLP
Download the [VERSION 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html).
and rewrite the variable `CORENLP_HOME` which indicates location of Stanford CoreNLP packages in `./setting.yaml`

URL:
http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

### Run Preprocess Steps
One can not remove the `sudo` in the following scripts, or the Stanford-NLP-server will report 403 forbidden error.
`
sudo python3 odee_preprocess_stanza.py $SCRIPTPATH/data/test_data parsed_test
`

bash:
```
sudo python3 odee_preprocess.py $SCRIPTPATH/data/test_data parsed_test
sudo python3 odee_preprocess.py $SCRIPTPATH/data/dev_data parsed_dev
sudo python3 odee_preprocess.py $SCRIPTPATH/data/unlabeled_data parsed_unlabeled
sudo chown $THIS_USER parsed_test
sudo chown $THIS_USER parsed_dev
sudo chown $THIS_USER parsed_unlabeled
```

### Copy Labeled Data
bash
```
python copy_labeled.py $SCRIPTPATH/data/test_data parsed_test
python copy_labeled.py $SCRIPTPATH/data/dev_data parsed_dev
```

## Prepare Full Text as Reference Corpus
bash
```
sudo python3 prepare_ref_corpus.py $SCRIPTPATH/data/test_data corpus.test
sudo python3 prepare_ref_corpus.py $SCRIPTPATH/data/dev_data corpus.dev
sudo python3 prepare_ref_corpus.py $SCRIPTPATH/data/unlabeled_data corpus.unlabeled
sudo chown $THIS_USER corpus.test
sudo chown $THIS_USER corpus.dev
sudo chown $THIS_USER corpus.unlabeled
cat corpus.test corpus.dev corpus.unlabeled > corpus
```

## Produced Data
1. `*.txt`: original text
2. `*.json`: json-style ODEE input
3. `corpus.*`: tokenized full text corpus
