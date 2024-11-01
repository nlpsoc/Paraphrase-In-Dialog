"""
    script to fine-tune the DeBERTa token classifier
"""
import argparse
import logging
import os
import sys

import pandas as pd
from transformers import TrainingArguments, Trainer

import eval_token_classifier
from paraphrase.utility.project_functions import set_logging
from paraphrase.set_id_consts import set_cache
from paraphrase.interview_data import MediaSumProcessor
from paraphrase.annotation_data import get_aggregated_annotations, DEV_PATH, TRAIN_PATH
from paraphrase.utility.stats import set_global_seed, SEED
import random
from paraphrase.token_classifier import TokenClassifier

MOCK_DATA = [  # generated
    {
        "sentence1": "I love coding in Python.".split(),
        "sentence2": "Python is a powerful programming language.".split(),
        "labels_sentence1": [0, 0, 1, 1, 0],  # Labels for each word in sentence1
        "labels_sentence2": [1, 1, 1, 1, 1, 1],  # Labels for each word in sentence2
    },
    {
        "sentence1": "Artificial intelligence will shape the future.".split(),
        "sentence2": "Machine learning is a subset of AI.".split(),
        "labels_sentence1": [1, 1, 1, 1, 1, 0],  # Labels for each word in sentence1
        "labels_sentence2": [1, 1, 1, 1, 1, 0, 0],  # Labels for each word in sentence2
    }
]


def main(model_id="roberta-base", save_path="../output/tokenclassification/", train_data="mock", test_data="mock",
         seed=SEED, test=False, batch=16, epochs=8, learning_rate=3e-5, to_hub=False, deduplicate=False):
    assert train_data in ["mock", "train"], "train_data must be one of 'mock' or 'train'"
    assert test_data in ["mock", "dev", "test"], "test_data must be one of 'mock' or 'dev'"
    assert learning_rate in [1e-5, 3e-5, 5e-5], "learning_rate must be one of 1e-5, 3e-5, 5e-5"
    set_logging()
    set_cache()
    set_global_seed(seed)
    # check if save path exists otherwise create
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.info(f"Saving results, checkpoints and _model to {save_path}")

    if (train_data == "train") or (test_data == "dev"):
        interview = MediaSumProcessor()

        logging.info("getting data. This might take a while ...")
        if train_data == "train":
            data_path = TRAIN_PATH
            result_data = get_individual_token_label_data(data_path, interview, test=test, deduplicate=deduplicate)
            train_data = result_data

        if test_data == "dev":
            data_path = DEV_PATH
            result_data = get_individual_token_label_data(data_path, interview, test=test, deduplicate=deduplicate)
            test_data = result_data

    if train_data == "mock":
        train_data = MOCK_DATA
    if test_data == "mock":
        test_data = MOCK_DATA

    logging.info("data fetched!")

    token_classifier = TokenClassifier(model_path=model_id)

    logging.info("Tokenizing data ...")
    train_data = token_classifier.tokenize_data(train_data)
    test_data = token_classifier.tokenize_data(test_data)
    data_collator = token_classifier.get_data_collator()
    logging.info("Tokenizing done!")

    logging.info("Training _model ...")

    logging.info(f"Saving results, checkpoints and _model to {save_path}")
    # https://huggingface.co/docs/transformers/en/tasks/token_classification#train
    training_args = TrainingArguments(
        output_dir=f"{save_path}results",
        # max_steps=steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch,
        logging_dir=f"{save_path}logs",
        load_best_model_at_end=True,
        evaluation_strategy="epoch",  # Set evaluation strategy to match the save strategy
        save_strategy="epoch",  # Set save strategy to match the evaluation strategy
    )

    # num_train_epochs=5,

    trainer = Trainer(
        model=token_classifier.model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=token_classifier.tokenizer,
        data_collator=data_collator,
        eval_dataset=test_data,
    )

    # Train the _model
    trainer.train()

    logging.info("Training done!")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.info(f"Saving _model to {save_path}")
    # save the _model
    token_classifier.save_model(save_path)

    eval_token_classifier.main(model_path=save_path, eval_set="dev")

    return token_classifier.model


def get_individual_token_label_data(data_path, interview, test=False, deduplicate=False):
    train_df = pd.read_csv(data_path, sep="\t", encoding="utf-16")
    # get single annotations for training
    annotation_dict = get_aggregated_annotations(train_df, interview, test=test)
    # shuffle the keys to randomize selection
    example_order = [i for i in range(len(annotation_dict["annotation_ids"]))]
    random.shuffle(example_order)
    # create the list of examples in same format as MOCK_DATA
    result_data = []
    for i in example_order:
        item = {"sentence1": annotation_dict["g_tokens"][i],
                "sentence2": annotation_dict["h_tokens"][i],
                "labels_sentence1": annotation_dict["g_weights"][i],
                "labels_sentence2": annotation_dict["h_weights"][i]}
        # check if item already exits
        if deduplicate and (item in result_data):
            continue
        result_data.append(item)
    del annotation_dict
    logging.info(f"Number of examples: {len(result_data)}")
    return result_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Token Classification Fine-Tuning Configurations')
    parser.add_argument('-_model', '--model_path', default="microsoft/deberta-v3-large", help="huggingface _model id")
    parser.add_argument('-train', '--train_set', default="train",
                        help="'train' or 'mock' ")
    parser.add_argument('-test', '--test_set', default="dev",
                        help="'dev' or 'mock' or 'test' ")
    parser.add_argument('-save', '--save_dir', help="path to where the _model should be saved")
    parser.add_argument('-seed', '--seed', help="path seed to be used")
    parser.add_argument('-batch', '--batch_size', default=16, help="batch size to be used")
    # parser.add_argument('-steps', '--steps', default=1000, help="number of steps to train the _model")
    parser.add_argument('-epochs', '--epochs', default=8, help="number of steps to train the _model")
    parser.add_argument('-lr', '--learning_rate', default=3e-5, help="number of steps to train the _model")
    # add bool for whether going in debug mode
    parser.add_argument('-debug', '--debug', default=False, help="whether to go in debug mode")
    parser.add_argument("--deduplicate", action="store_true", help="deduplicate the train data")

    args = parser.parse_args()

    main(model_id=args.model_path, train_data=args.train_set, test_data=args.test_set, save_path=args.save_dir,
         seed=int(args.seed), batch=int(args.batch_size), epochs=int(args.epochs),
         learning_rate=float(args.learning_rate), deduplicate=args.deduplicate)
