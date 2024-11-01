"""
    code to train and use a token classifier
"""
import logging
import os

import torch
import torch.nn.functional as F
# from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from paraphrase.utility.python_helpers import to_dict_of_lists


class TokenClassifier:
    def __init__(self, model_path: str):
        # CHECK IF directory exists
        if os.path.exists(model_path):
            logging.info(f"Loading model from {model_path}")
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        else:
            logging.info(f"Loading model from Huggingface model hub {model_path}")
            self.model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def tokenize_data(self, data_set, to_dataset=True):
        """
            expects data in the form

            [
                {
                    "sentence1": "I love coding in Python.",
                    "sentence2": "Python is a powerful programming language.",
                    "labels_sentence1": [0, 0, 1, 1, 0],  # Labels for each word in sentence1
                    "labels_sentence2": [1, 1, 1, 1, 1, 1],  # Labels for each word in sentence2
                }, ...
            ]

        :param to_dataset:
        :param data_set:
        :return:
        """
        # get data into form expected, see also https://huggingface.co/docs/transformers/en/tasks/token_classification
        data_set = self._to_SEP_format(data_set)
        if to_dataset:
            data_set = to_dict_of_lists(data_set)
            data_set = Dataset.from_dict(data_set)
            data_set = data_set.map(lambda x: self._tokenize_and_align_labels(x, padding=True), batched=True)
            return data_set
        else:
            return self._tokenize_and_align_labels(data_set, padding=True)

    def get_data_collator(self):
        return DataCollatorForTokenClassification(self.tokenizer)

    def inference(self, data_set, return_prob=False, pred_thresh=0.5):
        """

        :param data_set:
        :return:
        """
        # data_set = self.tokenize_data(data_set)

        data_set = self._to_SEP_format(data_set, labels=False)
        input_data = self._tokenize_and_align_labels(data_set, padding=True, labels=False)

        with torch.no_grad():
            outputs = self.model(input_data["input_ids"], input_data["attention_mask"])
        # Apply softmax to logits to get probabilities
        predicted_probs = F.softmax(outputs.logits, dim=-1)
        # predicted_labels = torch.argmax(outputs.logits, dim=2)

        # Decode the predicted labels back to words
        predicted_label_for_words = []
        probabilities_for_words = []
        for i in range(len(data_set)):
            cur_label_for_words = []
            cur_prob_for_words = []
            sep_index = data_set[i]["words"].index("[SEP]")
            word_index = input_data.word_ids(batch_index=i)
            for prob, word_index in zip(predicted_probs[i], word_index):  # label, predicted_labels[i],
                if (word_index is not None) and word_index == len(cur_label_for_words):  # use prediction for the first token of a given word
                    cur_label_for_words.append(1 if prob[1].item() > pred_thresh else 0)
                    cur_prob_for_words.append(prob[1].item())  # second entry is the probability for label 1
            predicted_label_for_words.append([cur_label_for_words[:sep_index], cur_label_for_words[sep_index+1:]])
            probabilities_for_words.append([cur_prob_for_words[:sep_index], cur_prob_for_words[sep_index+1:]])

        if not return_prob:
            return predicted_label_for_words
        else:
            return predicted_label_for_words, probabilities_for_words

    @staticmethod
    def _to_SEP_format(data_items, labels=True):
        """

            converts dataset tp the form
                [
                    {
                        "words": ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
                        "labels": [0, 0, 1, 0, 0, 0, 1, 1, 0]
                    }, ...
                ]

            starting from

                [
                    {
                        "sentence1": "I love coding in Python.",
                        "sentence2": "Python is a powerful programming language.",
                        "labels_sentence1": [0, 0, 1, 1, 0],  # Labels for each word in sentence1
                        "labels_sentence2": [1, 1, 1, 1, 1, 1],  # Labels for each word in sentence2
                    }, ...
                ]
        :param data_items:
        :return:
        """

        result_data = []
        for item in data_items:
            words = item["sentence1"] + ["[SEP]"] + item["sentence2"]
            if labels:
                labels = item["labels_sentence1"] + [0] + item["labels_sentence2"]
                result_data.append({"words": words, "labels": labels})
            else:
                result_data.append({"words": words})

        return result_data

    def _tokenize_and_align_labels(self, examples, padding=False, labels=True):
        """

        :param examples:
        :param tokenizer:
        :return:
        """
        if type(examples) == list:
            # reformat data from list of dicts to dict of lists
            examples = {k: [d[k] for d in examples] for k in examples[0]}

        tokenized_inputs = self.tokenizer(examples["words"], truncation=True, is_split_into_words=True,
                                          return_tensors="pt", padding=padding)

        if labels:
            labels = []
            for i, label in enumerate(examples[f"labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        try:
                            label_ids.append(label[word_idx])
                        except IndexError:
                            print(f"word_idx: {word_idx}, label: {label}, i: {i}")
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
        return tokenized_inputs


