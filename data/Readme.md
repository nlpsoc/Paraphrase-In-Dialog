 You only need this if you want to replicate our annotation approach. Otherwise check out https://huggingface.co/datasets/AnnaWegmann/Paraphrases-in-Interviews for our provided dataset. 
 
# Adding the original Interview Data

You probably don't need this, rather use `result/Data/cut_news_dialogue.tsv` (see below)

Download `news_dialog.zip` via: https://github.com/zcgzcgzcg1/MediaSum

De-compress and put `news_dialog.json` (4.5 GB) in `MediaSum` directory. We use the same IDs for the interviews in our data as in the original `news_dialog.json`.

Data format of `news_dialog.json`:

```json
{
  "id": "NPR-11",
  "program": "Day to Day",
  "date": "2008-06-10",
  "url": "https://www.npr.org/templates/story/story.php?storyId=91356794",
  "title": "Researchers Find Discriminating Plants",
  "summary": "The \"sea rocket\" shows preferential treatment to plants that are its kin. Evolutionary plant ecologist Susan Dudley of McMaster University in Ontario discusses her discovery.",
  "utt": [
    "This is Day to Day.  I'm Madeleine Brand.",
    "And I'm Alex Cohen.",
    "Coming up, the question of who wrote a famous religious poem turns into a very unchristian battle.",
    "First, remember the 1970s?  People talked to their houseplants, played them classical music. They were convinced plants were sensuous beings and there was that 1979 movie, \"The Secret Life of Plants.\"",
    "Only a few daring individuals, from the scientific establishment, have come forward with offers to replicate his experiments, or test his results. The great majority are content simply to condemn his efforts without taking the trouble to investigate their validity.",
    ...
    "OK. Thank you.",
    "That's Susan Dudley. She's an associate professor of biology at McMaster University in Hamilt on Ontario. She discovered that there is a social life of plants."
  ],
  "speaker": [
    "MADELEINE BRAND, host",
    "ALEX COHEN, host",
    "ALEX COHEN, host",
    "MADELEINE BRAND, host",
    "Unidentified Male",    
    ..."
    Professor SUSAN DUDLEY (Biology, McMaster University)",
    "MADELEINE BRAND, host"
  ]
}
```



## the relevant sub-sample (result/Data/cut_news_dialogue.tsv)

see `result/Data/cut_news_dialogue.tsv` which contains the random subsample (1304 interviews) considered in our paper.
