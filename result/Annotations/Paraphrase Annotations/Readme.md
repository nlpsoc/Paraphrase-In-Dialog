In case you want to use our code for evaluation and/or loading you need this data format.

## Data Files

All crowd-sourced annotations are in `all_annotations.tsv`

See the folder   
`5050` for the original qualtrics files for the 50/50 set and  
`RAND-PARA` for the original qualtrics files for the RANDOM/PARA sets.

`dev.tsv` includes dev set annotations  
`test.tsv` includes test set annotations  
`train.tsv` includes train set annotations

To use the annotation data you can immediately load the tsv file with probably all information you want, see `ANON_ALL-TOKEN-annotations.tsv`. If you want to use the data split, load`ANON_TRAIN-TOKEN-annotations.tsv`, `ANON_DEV-TOKEN-annotations.tsv` and `ANON_TEST-TOKEN-annotations.tsv`  in the folder `result/Annotations/Paraphrase Annotations`. Datasets are of the below form. 

"Guest HLs" indicates the proportion of annotators that highlighted the nth word (same order as "Guest Tokens"), only considering those who classified the pair as a paraphrase. 
"Vote" includes the number of annotators classifying the pair as a paraphrase out of the total number of annotators. For example [10, 20] means 10 out of 20 annotators classified the pair as a paraphrase.

| QID          | Guest Tokens                                                 | Guest HLs                         | Host Tokens                                                  | Host HLs                                              | Vote     |
| ------------ | ------------------------------------------------------------ | --------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- | -------- |
| CNN-177596-7 | ['This', 'is', 'not', 'good.']                               | [1.0, 0.9, 0.9, 0.9]              | ['This', 'is', 'what', 'you', "don't", 'want', 'happening', 'with', 'your', 'menorah,',... | [0.9, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7,... | [10, 20] |
| NPR-8678-6   | ['Well,', 'earlier', 'this', 'month,', "Guatemala's", 'highest', 'court', 'had', 'blocked', 'the',... | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,... | ['Did', 'President', "Trump's", 'threat', 'to', 'the', 'Guatemalan', 'leadership', 'and', 'government',... | [0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.4, 0.4,... | [5, 20]  |

or you can use the code:

```
from paraphrase.annotation_data import get_human_anns

guest_tokens_per_qid, host_tokens_per_qid, human_anns_per_qid, human_class_per_qid = (
        get_human_anns())  # question_ids="DEV"/"TEST"/"TRAIN" for the splits
```



## Data Format

Annotations are given in the following form

| QID               | Category     | Highlighted                                          | Annotator  | Session           | Session_Start       |
| ----------------- | ------------ | ---------------------------------------------------- | ---------- | ----------------- | ------------------- |
| R_2PoEZfAptrkFdsx | Comment      | Some of these were hard. I felt like the majority... | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| R_2PoEZfAptrkFdsx | Time         | 1026                                                 | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| R_2PoEZfAptrkFdsx | # failed ACs | 0                                                    | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| R_2PoEZfAptrkFdsx | EndTime      | 2023-06-14 07:25:40                                  | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-177596-7      | Referred     | []                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-177596-7      | Paraphrase   | []                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-177596-7      | Is Referring | No                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-177596-7      | Comment      | no                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| NPR-8678-6        | Referred     | []                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| NPR-8678-6        | Paraphrase   | []                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| NPR-8678-6        | Is Referring | No                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| NPR-8678-6        | Comment      | no                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-350097-7      | Referred     | [2, 3, 4, 5]                                         | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-350097-7      | Paraphrase   | [4, 5, 6, 7, 8, 9]                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-350097-7      | Is Referring | Yes                                                  | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-350097-7      | Comment      | no                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-235909-21     | Referred     | []                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-235909-21     | Paraphrase   | []                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-235909-21     | Is Referring | No                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |
| CNN-235909-21     | Comment      | no                                                   | PROLIFIC_1 | R_2PoEZfAptrkFdsx | 2023-06-14 07:08:33 |

i.e., the relevant items are those starting with "CNN" or "NPR" (representing the ID of the item), where an annotator (Annotator column) annotated a paraphrase if Referred and Paraphrase both include list of ints, those represent the highlights done by the annotator in the guest (Referred) and host (Paraphrase) utterance.