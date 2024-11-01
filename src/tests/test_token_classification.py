from unittest import TestCase

from transformers import AutoTokenizer

import paraphrase.token_classifier as token_classifier
import token_classifier

import paraphrase.qualtrics
import paraphrase.utility.qualtrics_api
import paraphrase.utility.qualtrics_survey


class Test(TestCase):

    def setUp(self):
        self.huggingface_format = [{'words': ['I', 'love', 'coding', 'in', 'Python.', '[SEP]', 'Python', 'is', 'a',
                                              'powerful', 'programming', 'language.'],
                                    'labels': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]},
                                   {'words': ['Artificial', 'intelligence', 'will', 'shape', 'the', 'future.', '[SEP]',
                                              'Machine', 'learning', 'is', 'a', 'subset', 'of', 'AI.'],
                                    'labels': [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0]}]

    def test_to_huggingface_format(self):
        newformat = token_classifier.to_SEP_format(
            token_classification.MOCK_DATA)
        self.assertEqual(newformat, self.huggingface_format)

    def test_align_labels(self):
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        tokenized_data = token_classifier.tokenize_and_align_labels(self.huggingface_format[0], tokenizer)
        print(tokenizer.convert_ids_to_tokens(tokenized_data["input_ids"][0]))
        print(tokenized_data)

    def test_dataset_map(self):
        from datasets import IterableDataset
        ds = IterableDataset.from_generator(token_classification.LIST_GENERATOR)
        tokenized_data = ds.map(token_classifier.tokenize_and_align_labels, batched=True)
        print(tokenized_data)

        # tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def test_train_token_classification(self):
        paraphrase.qualtrics.create_survey(model_id="distilbert/distilbert-base-uncased")

    def test_train_with_annotation_data(self):

        # load train annotation data
        paraphrase.qualtrics.create_survey(model_id="distilbert/distilbert-base-uncased", train_data="train", test_data="dev",
                                           test=True, batch=2, epochs=1)

    def test_token_classifier_from_hub(self):
        model = token_classifier.TokenClassifier("AnnaWegmann/Paraphrase-In-Dialog")
        model.inference(self.huggingface_format)
