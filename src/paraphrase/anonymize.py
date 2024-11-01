"""
    path to replace prolific IDs with anonymized IDs
"""
import ast
import logging
import os

import pandas as pd

from paraphrase.set_id_consts import LEAD
from paraphrase.utility.annotation_df import QualtricsColumns, AnnotationColumns
from paraphrase.utility.annotation_pipeline import QualtricsProlificDB
from paraphrase.utility.project_functions import get_dir_to_src

ANON_LEAD = "Lead"
TRANSLATION_PATH = get_dir_to_src() + "/../result/Anonymize/annotator_id_translations.tsv"


class IDTranslationColumns:
    Original_ID = "Original_ID"
    Anon_ID = "Anon_ID"


def anonymize_folder(folder_path, id_translation_path=TRANSLATION_PATH):
    #   get all tsv files in the folder
    tsv_files = [f for f in os.listdir(folder_path) if f.endswith(".tsv")]
    #   anonymize each tsv file
    for tsv_file in tsv_files:
        tsv_file_path = os.path.join(folder_path, tsv_file)
        if "ANON" in tsv_file:  # do not already anonymized files
            continue
        try:
            anonymize_file(tsv_file_path, id_translation_path)
        except ValueError:
            print(f"Skipping unexpected file type {tsv_file_path}")
    return


def anonymize_file(tsv_file_path, id_translation_path=TRANSLATION_PATH):
    try:
        df = pd.read_csv(tsv_file_path, sep="\t", encoding="utf-16")
    except UnicodeError:
        logging.info("Reading file with utf-8 encoding")
        df = pd.read_csv(tsv_file_path, sep="\t")

    if QualtricsColumns.Author_ID in df.columns:
        id_column = QualtricsColumns.Author_ID
    elif AnnotationColumns.Annotator_ID in df.columns:
        id_column = AnnotationColumns.Annotator_ID
    elif QualtricsProlificDB.ADMITTED in df.columns:
        # DB file
        id_column = [QualtricsProlificDB.ADMITTED, QualtricsProlificDB.PARTICIPATED, QualtricsProlificDB.VALID]
    else:
        raise ValueError("No ID column found in the dataframe")

    # UPDATE THE TRANSLATION FILE
    id_translation = _update_translation_df(df, id_translation_path, id_column)

    # ANONYMIZE THE DATAFRAME
    _anonymize_df(df, id_translation, tsv_file_path, id_column)

    return


def _anonymize_df(df, id_translation, tsv_file_path, id_column):
    #   create a copy of the original dataframe df with the anonymized ids
    anon_df = df.copy()

    replacement_dict = dict(
        zip(id_translation[IDTranslationColumns.Original_ID], id_translation[IDTranslationColumns.Anon_ID]))
    if type(id_column) == list:  # DB file
        for col in id_column:
            if col == QualtricsProlificDB.ADMITTED:  # ignore the TRAIN data for Admitted annotators
                anon_df.loc[anon_df["Version"] != "TRAIN", col] = anon_df[anon_df["Version"] != "TRAIN"][col].apply(
                    lambda lst: [replacement_dict[item] if item in replacement_dict else item
                                 for item in ast.literal_eval(str(lst))])
            else:
                anon_df[col] = anon_df[col].apply(
                    lambda lst: [replacement_dict[item] for item in ast.literal_eval(str(lst))])

        # get admitted annotator ids and participated (they should be superset of valid)
        # anon_df[id_column] = anon_df[id_column].applymap(
        #     lambda lst: [
        #         [id_translation[id_translation[IDTranslationColumns.Original_ID] == item][IDTranslationColumns.Anon_ID].values[0] for item in lst[col]]
        #         for col in id_column]
        # )
    else:
        #   update the Author_ID column with the anonymized ids
        anon_df[id_column] = anon_df[id_column].replace(replacement_dict)
    # save the anonymized dataframe starting with "ANON_"
    #   get the filename without the path
    filename = os.path.basename(tsv_file_path)
    #   add "ANON_" to the filename
    anon_filename = f"ANON_{filename}"
    #   prepend the path to the filename
    anon_path = os.path.join(os.path.dirname(tsv_file_path), anon_filename)
    anon_df.to_csv(anon_path, sep="\t", encoding="utf-16", index=False)


def _update_translation_df(annotation_df, id_translation_path, id_column):
    #   read in id translation file, create it if it does not exist
    try:
        id_translation = pd.read_csv(id_translation_path, sep="\t")
    except FileNotFoundError:
        id_translation = pd.DataFrame(columns=[IDTranslationColumns.Original_ID, IDTranslationColumns.Anon_ID])
    #   get the unique original ids in the translation file (i.e., those that already have a translation)
    unique_ids = id_translation[IDTranslationColumns.Original_ID].unique()

    #   get the maximum prolific id in the translation file
    max_prolific_id = 0
    for a_id in id_translation[IDTranslationColumns.Anon_ID].unique():
        if a_id.startswith("PROLIFIC_"):
            id_num = int(a_id.split("_")[1])
            if id_num > max_prolific_id:
                max_prolific_id = id_num

    #   get all unique ids in the tsv file
    if type(id_column) == list:  # DB file
        # get admitted annotator ids and participated (they should be superset of valid)
        unique_ids = sum(annotation_df[annotation_df["Version"] != "TRAIN"][id_column[0]].apply(ast.literal_eval),
                         [])
        unique_ids += sum(annotation_df[id_column[1]].apply(ast.literal_eval), [])
        unique_ids = list(set(unique_ids))
    elif id_column == QualtricsColumns.Author_ID:  # "Q-PID" is only used in Qualtrics files
        unique_ids = annotation_df.iloc[2:][id_column].unique()  # ignore first two rows of metadata
    else:
        unique_ids = annotation_df[id_column].unique()

    #   create anonymized version of the form "PROLIFIC_1", "PROLIFIC_2", etc.
    app_original_ids = []
    app_anon_ids = []
    for a_id in unique_ids:
        # check if the id is not given
        if pd.isna(a_id):
            continue  # skip nan authors, they are already anonymized
        if (a_id not in id_translation[IDTranslationColumns.Original_ID].values) and (a_id not in app_original_ids):
            if LEAD in a_id.lower():
                # replace casing-ignorant lead author name with "Annotator"
                an_id = a_id.lower().replace(LEAD, ANON_LEAD)
            else:
                an_id = f"PROLIFIC_{max_prolific_id + 1}"
                max_prolific_id += 1
            app_original_ids.append(a_id)
            app_anon_ids.append(an_id)
    #   update the translation df
    app_df = pd.DataFrame({IDTranslationColumns.Original_ID: app_original_ids,
                           IDTranslationColumns.Anon_ID: app_anon_ids})
    id_translation = pd.concat([id_translation, app_df])
    # save the updated translation file
    id_translation.to_csv(id_translation_path, sep="\t", index=False)
    return id_translation
