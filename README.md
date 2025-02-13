# Open Domain Event Extraction Using Neural Latent Variable Models (ODEE)
This is the python3 code for the paper ["Open Domain Event Extraction Using Neural Latent Variable Models"](https://arxiv.org/abs/1906.06947) in ACL 2019.

## Prepare ELMo model
Modify the Line 24 and 25 in `cache_features.py`.

The fine-tune process need 2 * GTX 1080Ti, if the fine-tune process is costly or somehow failed 
to complete, please use the initial parameters in [allennlp](https://allennlp.org/elmo).

Using the "small" version of ElMo.
https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5
https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json

Please note that it is optional to finetune the ELMo model if you just want to complete the whole procedure
or use the model in somewhere else.

## Prepare Data and Train Model

The data is [HERE](https://drive.google.com/open?id=1KjL3mAxj9nmzqC75s2rNaT6x6CJBZZTj).

1. run the dataprocessor
2. run `sudo chown [YOUR_UERS] [PROCESSED_DIR]` and specify the directories in `setting.yaml` manually
2. run `pip3 install -r requirements.txt` to install required packages
3. run `python3 cache_features.py`
4. run `python3 train_avitm.py 2>&1 | tee logs/train.log`
5. run `python3 generate_slot_topN.py 2>&1 | tee logs/generation.log`
6. run `python3 decode.py 2>&1 | tee logs/decoding.log`
7. follow steps in `slotcoherence/README.md` to split the corpus
7. run `cd slotcoherence && bash ./run-oc.sh 2>&1 | tee $SCRIPTPATH/logs/coherence.log && cd ..`
8. run `visualize_test.ipynb`

## Produced Data
1. `*.json.pt`: cached features of ODEE input
2. `*.json.answer`: decoded full results of a news group
3. `*.json.template`: decoded template of a news group
4. `*.json.events.topN`: decoded top-N events of a news group
5. `*.json.labeled`: labeled events of test split
6. `slotcoherence/slot_head_words.txt`: generated topN head words for each slot

# Results

After about 80 epoches (2021-05-13):
```
Average Topic Coherence = 0.076
Median Topic Coherence = 0.075
```

After about 110 epoches (2021-05-17-4), loss ~ 4824:
```
Average Topic Coherence = 0.133
Median Topic Coherence = 0.119
```

After about 130 epoches (2021-05-17-2), loss ~ 4795:
```
Average Topic Coherence = 0.141
Median Topic Coherence = 0.155
```

After about 150 epoches (2021-05-17-3), loss ~ 4785:
```
Average Topic Coherence = 0.139
Median Topic Coherence = 0.126
```

After about 260 epoches (2021-05-14):
```
Average Topic Coherence = 0.109
Median Topic Coherence = 0.104
```

After about 460 epoches (2021-05-17), loss ~ 4752:
```
Average Topic Coherence = 0.074
Median Topic Coherence = 0.067
```

![image](chi/slot-coherence.png)

## Cite
Please cite our ACL 2019 paper:
```bibtex
@inproceedings{DBLP:conf/acl/LiuHZ19,
  author    = {Xiao Liu and
               Heyan Huang and
               Yue Zhang},
  title     = {Open Domain Event Extraction Using Neural Latent Variable Models},
  booktitle = {Proceedings of the 57th Conference of the Association for Computational
               Linguistics, {ACL} 2019, Florence, Italy, July 28- August 2, 2019,
               Volume 1: Long Papers},
  pages     = {2860--2871},
  year      = {2019},
  crossref  = {DBLP:conf/acl/2019-1},
  url       = {https://www.aclweb.org/anthology/P19-1276/},
  timestamp = {Wed, 31 Jul 2019 17:03:52 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/acl/LiuHZ19},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
