`Annotations` includes the human annotations on our dataset in the format that our code expects. You probably don't want to use this though. Use the dataset on huggingface instead https://huggingface.co/datasets/AnnaWegmann/Paraphrases-in-Interviews. 

`Anonymize` includes the translation between prolific IDs and the pseudos like PROLIFIC_1 (removed for public release).

`Data` includes the sample of the original MediaSum interview dataset that we considered for annotation.

`Models` includes the in-context learning prompts and responses on the dev and test sets. For our trained DeBERTa models see https://huggingface.co/AnnaWegmann/Highlight-Paraphrases-in-Dialog-ALL for ALL model and https://huggingface.co/AnnaWegmann/Highlight-Paraphrases-in-Dialog for AGGREGATE model, see https://arxiv.org/abs/2404.06670.