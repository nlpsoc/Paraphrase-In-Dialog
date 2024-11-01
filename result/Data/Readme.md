# Sampled Dataset

See the used subsample of MediaSum  (1304 interviews)  in the paper in `result/Data/cut_news_dialogue.tsv`. This is after preprocessing and sampling but before annotation.

Contains Data of the form:

```
| id | program | date | url | title | summary | utt | speaker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NPR-58 | Weekend Edition Sunday | 2016-10-02 | https://www.npr.org/2016/10/02/496282796/how-if-at-all-will-trumps-tax-records-sway-the-race | How, If At All, Will Trump's Tax Records Sway The... | How will Trump respond to his 1995 tax records being... | ["We're joined now by NPR national political correspondent Mara Liasson... | ['RACHEL MARTIN, HOST', 'MARA LIASSON, BYLINE', 'RACHEL MARTIN, HOST', 'MARA... |
| NPR-217 | Day to Day | 2008-05-26 | https://www.npr.org/templates/story/story.php?storyId=90828052 | Fate of Michigan, Florida Delegates on the Table | Uncounted delegates from Michigan and Florida are again in question... | ["From the studios of NPR West, this is Day to... | ['ALEX CHADWICK, host', 'ALEX CHADWICK, host', 'RON ELVING', 'ALEX CHADWICK,... |

```





## Preprocessing

The `preprocessing` folder includes logs and results of the preprocessing step for the MediaSum data.

`ieir_ids.csv` is the tsv file of the preprocessed dataset of (created in `preprocess_interviews.py`) with the (guest, host) utterances ordered randomly. ".csv" ending is a legacy reason, really it should be a ".tsv" ending.

Note: `ieir_ids.csv` doesn't yet include the filtering with max tokens of length 200. This limit was introduced later with practical restrictions for highlight questions in qualtrics of 200 tokens. In the paper we report preprocessing stats AFTER removal of 200 token (g, h)-pairs.







