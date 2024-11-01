"""
    You probably don't want to use this

    Testing creation of annotation surveys

"""
from unittest import TestCase

import pandas as pd

from create_annotation_surveys import main


class Test(TestCase):
    def test_train_creation(self):
        df = pd.DataFrame(columns=["Survey Name", "Qualtrics ID", "Prolific ID", "Version", "STATUS",
                                   "Question IDs", "Admitted Annotators", "Participated Annotators",
                                   "Valid Annotators", "Entropy"])
        updated_df = main(an_type="TRAINING", test=True, test_df=df, places=4)
        self.assertEqual(len(updated_df), 1)
        """
            Survey Name	                                    Qualtrics ID	Prolific ID	    Version	    STATUS	    
            TEST_DATE_Training_Paraphrase-Highlighting      SV_surveyid     65prolificid    TRAIN       INIT
            
            Question IDs	Admitted Annotators	    Participated Annotators	    Valid Annotators	Entropy
            -               4                       ""                          ""                  -
        """
        self.assertRegex(updated_df["Survey Name"].iloc[0], "^TEST_.*Training_Paraphrase-Highlighting")
        self.assertRegex(updated_df["Qualtrics ID"].iloc[0], "^SV_.*")
        self.assertRegex(updated_df["Prolific ID"].iloc[0], "^65.*")
        self.assertEqual(updated_df["Version"].iloc[0], "TRAIN")
        self.assertEqual(updated_df["STATUS"].iloc[0], "INIT")
        self.assertEqual(updated_df["Question IDs"].iloc[0], "-")
        self.assertEqual(updated_df["Entropy"].iloc[0], "-")
        self.assertEqual(updated_df["Admitted Annotators"].iloc[0], 4)
        self.assertEqual(updated_df["Participated Annotators"].iloc[0], "")
        self.assertEqual(updated_df["Valid Annotators"].iloc[0], "")

    def test_annotator_init(self):
        value_dict = {
            "Survey Name": ["TEST_23-09-29_Training_Paraphrase-Highlighting"],
            "Qualtrics ID": ["SV_IMADETHISUP"],
            "Prolific ID": ["6516eeIMADETHISUP"],
            "Version": ["TRAIN"],
            "STATUS": ["INIT"],
            "Question IDs": ["-"],
            "Admitted Annotators": [3],
            "Participated Annotators": [""],
            "Valid Annotators": [""],
            "Entropy": ["-"]
        }
        df = pd.DataFrame(value_dict)
        updated_df = main(an_type="ANNOTATOR-INIT", test=True, test_df=df)
        self.assertEqual(len(updated_df), 2)
        value_dict["STATUS"] = ["COMPLETE"]
        value_dict["Participated Annotators"] = [['Annotator1', 'Annotator2', 'Annotator3']]
        value_dict["Valid Annotators"] = [['Annotator1', 'Annotator2', 'Annotator3']]
        self.assertTrue(updated_df.iloc[0].equals(pd.DataFrame(value_dict).iloc[0]))
        """
            
        """
        self.assertRegex(updated_df["Survey Name"].iloc[1], "^TEST_.*Paraphrase-Annotation_0-9")
        self.assertRegex(updated_df["Qualtrics ID"].iloc[1], "^SV_.*")
        self.assertRegex(updated_df["Prolific ID"].iloc[1], "^65.*")
        self.assertEqual(updated_df["Version"].iloc[1], 0)
        self.assertEqual(updated_df["STATUS"].iloc[1], "INIT")
        self.assertEqual(updated_df["Question IDs"].iloc[1],
                         ['CNN-177596-7', 'NPR-8678-6', 'CNN-350097-7', 'CNN-235909-21', 'NPR-23442-3', 'CNN-323090-3',
                          'CNN-72706-11', 'CNN-378275-5', 'CNN-300212-13', 'CNN-376903-8'])
        self.assertEqual(updated_df["Entropy"].iloc[1], None)
        self.assertSetEqual(set(updated_df["Admitted Annotators"].iloc[1]), set(['Annotator1', 'Annotator2', 'Annotator3']))
        self.assertEqual(updated_df["Participated Annotators"].iloc[1], None)
        self.assertEqual(updated_df["Valid Annotators"].iloc[1], None)

    def test_annotator_update(self):
        value_dict = {
            "Survey Name": ["TEST_23-09-29_Training_Paraphrase-Highlighting",
                            "TEST_23-10-09_Training_Paraphrase-Highlighting",
                            "TEST_23-10-10_Training_Paraphrase-Highlighting",
                            "TEST_2023-09-29_Paraphrase-Annotation_0-9"],
            "Qualtrics ID": ["SV_IMADETHISUP", "SV_SV_IMADETHISUP2", "SV_SV_IMADETHISUP3", "SV_4IM74gZtPIjWS1g"],
            "Prolific ID": ["6516eeIMADETHISUP", "6516eeIMADETHISUP2", "6516eeIMADETHISUP3",
                            "6516f6074831c4928917a416"],
            "Version": ["TRAIN", "TRAIN", "TRAIN", "0"],
            "STATUS": ["COMPLETE", "COMPLETE", "COMPLETE", "INIT"],
            "Question IDs": ["-", "-", "-",
                             "['CNN-177596-7', 'NPR-8678-6', 'CNN-350097-7', 'CNN-235909-21', 'NPR-23442-3', "
                             "'CNN-323090-3', 'CNN-72706-11', 'CNN-378275-5', 'CNN-300212-13', 'CNN-376903-8']"],
            "Admitted Annotators": ["3", "4", "4", ['Annotator2', 'Annotator3', 'Annotator1']],
            "Participated Annotators": [['Annotator1', 'Annotator2', 'Annotator3'], ['Annotator0', 'Annotator4', 'Annotator5', 'Annotator6'],
                                        ['Annotator11', 'Annotator8', 'Annotator9', 'Annotator10'], None],
            "Valid Annotators": ["['Annotator1', 'Annotator2', 'Annotator3']", "[ 'Annotator4', 'Annotator5', 'Annotator6']",
                                 "['Annotator8', 'Annotator9', 'Annotator10']", None],
            "Entropy": ["-", "-", "-", None]
        }
        df = pd.DataFrame(value_dict)
        updated_df = main(an_type="ANNOTATOR-UPDATE", test=True, test_df=df)
        print(updated_df)
        self.assertEqual(updated_df["STATUS"].iloc[3], "COMPLETE")
        self.assertEqual(updated_df["Valid Annotators"].iloc[3], ['Annotator1', 'Annotator2', 'Annotator3'])
        self.assertEqual(updated_df["Entropy"].iloc[3],
                         [[2.0, 3], [0.0, 3], [3.0, 3], [3.0, 3], [1.0, 3],
                          [1.0, 3], [0.0, 3], [3.0, 3], [3.0, 3], [3.0, 3]])
        self.assertEqual(updated_df["Question IDs"].iloc[4], ['CNN-177596-7', 'NPR-23442-3', 'CNN-323090-3'])
        self.assertEqual(updated_df["Version"].iloc[4], 1)
        self.assertEqual(updated_df["STATUS"].iloc[4], "INIT")
        self.assertSetEqual(set(updated_df["Admitted Annotators"].iloc[4]),
                            set(['Annotator6', 'Annotator4', 'Annotator5', 'Annotator9', 'Annotator10', 'Annotator8']))

    def test_annotator_last(self):
        value_dict = {
            "Survey Name": ["TEST_23-09-29_Training_Paraphrase-Highlighting",
                            "TEST_23-10-09_Training_Paraphrase-Highlighting",
                            "TEST_23-10-10_Training_Paraphrase-Highlighting",
                            "TEST_2023-09-29_Paraphrase-Annotation_0-9",
                            "TEST_2023-09-29_Paraphrase-Annotation_0-9",
                            "TEST_2023-09-29_Paraphrase-Annotation_0-9",
                            "TEST_2023-09-29_Paraphrase-Annotation_0-9"],
            "Qualtrics ID": ["SV_IMADETHISUP", "SV_SV_IMADETHISUP2", "SV_SV_IMADETHISUP3",
                             "SV_4IM74gZtPIjWS1g", "SV_097ANu2TQcBu6IS", "SV_00Uc0KWXv5SyXz0", "SV_0TcPsvQb3QzyXHw"],
            "Prolific ID": ["6516eeIMADETHISUP", "6516eeIMADETHISUP2", "6516eeIMADETHISUP3",
                            "6516f6074831c4928917a416", "652409a615c246ddfe764c27",
                            "65241a7aab0a84107aa4a6f5", "65241b6ea745597359c8051f"],
            "Version": ["TRAIN", "TRAIN", "TRAIN", "0", "1", "2", "3"],
            "STATUS": ["COMPLETE", "COMPLETE", "COMPLETE", "COMPLETE", "COMPLETE", "COMPLETE", "INIT"],
            "Question IDs": ["-", "-", "-",
                             "['CNN-177596-7', 'NPR-8678-6', 'CNN-350097-7', 'CNN-235909-21', 'NPR-23442-3', "
                             "'CNN-323090-3', 'CNN-72706-11', 'CNN-378275-5', 'CNN-300212-13', 'CNN-376903-8']",
                             ['CNN-177596-7', 'NPR-23442-3', 'CNN-323090-3'],
                             ['CNN-323090-3'],
                             ['CNN-323090-3']
                             ],
            "Admitted Annotators": ["3", "4", "4",
                                    ['Annotator2', 'Annotator3', 'Annotator1'], ['Annotator4', 'Annotator5'], ['Annotator6'],
                                    ['Annotator8', 'Annotator10', 'Annotator9']],
            "Participated Annotators": [['Annotator1', 'Annotator2', 'Annotator3'], ['Annotator0', 'Annotator4', 'Annotator5', 'Annotator6'],
                                        ['Annotator11', 'Annotator8', 'Annotator9', 'Annotator10'],
                                        ['Annotator1', 'Annotator2', 'Annotator3'], ['Annotator4', 'Annotator5'], ['Annotator6'], None],
            "Valid Annotators": ["['Annotator1', 'Annotator2', 'Annotator3']", "[ 'Annotator4', 'Annotator5', 'Annotator6']",
                                 "['Annotator8', 'Annotator9', 'Annotator10']",
                                 ['Annotator1', 'Annotator2', 'Annotator3'], ['Annotator4', 'Annotator5'], ['Annotator6'], None],
            "Entropy": ["-", "-", "-",
                        [[2.0, 3], [0.0, 3], [3.0, 3], [3.0, 3], [1.0, 3], [1.0, 3], [0.0, 3], [3.0, 3], [3.0, 3],
                         [3.0, 3]], [[4.0, 5], [1.0, 5], [2.0, 5]], [[2.0, 6]], None]
        }
        df = pd.DataFrame(value_dict)
        updated_df = main(an_type="ANNOTATOR-UPDATE", test=True, test_df=df)
        # print(updated_df)
        self.assertEqual(updated_df["Entropy"].iloc[6], [[2.0, 9]])
        self.assertEqual(updated_df["Version"].iloc[6], "LAST")
        self.assertEqual(updated_df["STATUS"].iloc[6], "COMPLETE")
        self.assertSetEqual(set(updated_df["Participated Annotators"].iloc[6]),
                            set(['Annotator8', 'Annotator10', 'Annotator9']))

    def test_remove_lowq_annotator(self):
        value_dict = {
            "Survey Name": ["TEST_23-09-29_Training_Paraphrase-Highlighting",
                            "TEST_23-10-09_Training_Paraphrase-Highlighting",
                            "TEST_23-10-10_Training_Paraphrase-Highlighting",
                            "TEST_2023-09-29_Paraphrase-Annotation_10-19",
                            "TEST_2023-09-29_Paraphrase-Annotation_20-29",
                            "TEST_2023-09-29_Paraphrase-Annotation_30-39",
                            "TEST_2023-09-29_Paraphrase-Annotation_40-49",
                            "TEST_2023-09-29_Paraphrase-Annotation_0-9"],
            "Qualtrics ID": ["SV_IMADETHISUP", "SV_SV_IMADETHISUP2", "SV_SV_IMADETHISUP3",
                             "SV_SV_IMADETHISUP4", "SV_SV_IMADETHISUP5", "SV_SV_IMADETHISUP6", "SV_SV_IMADETHISUP7",
                             "SV_4IM74gZtPIjWS1g"],
            "Prolific ID": ["6516eeIMADETHISUP", "6516eeIMADETHISUP2", "6516eeIMADETHISUP3",
                            "6516eeIMADETHISUP4", "6516eeIMADETHISUP5", "6516eeIMADETHISUP6", "6516eeIMADETHISUP7",
                            "6516f6074831c4928917a416"],
            "Version": ["TRAIN", "TRAIN", "TRAIN", "LAST", "LAST", "LAST", "LAST", "0"],
            "STATUS": ["COMPLETE", "COMPLETE", "COMPLETE",
                       "COMPLETE", "COMPLETE", "COMPLETE", "COMPLETE",
                       "INIT"],
            "Question IDs": ["-", "-", "-",
                             "[NPR-TEST-1]",  "[NPR-TEST-2]",  "[NPR-TEST-3]",  "[NPR-TEST-4]",
                             "['CNN-177596-7', 'NPR-8678-6', 'CNN-350097-7', 'CNN-235909-21', 'NPR-23442-3', "
                             "'CNN-323090-3', 'CNN-72706-11', 'CNN-378275-5', 'CNN-300212-13', 'CNN-376903-8']",
                             ],
            "Admitted Annotators": ["3", "4", "4",
                                    ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                    ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                    ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                    ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                    ['Annotator2', 'Annotator3', 'Annotator1']],
            "Participated Annotators": [['Annotator1', 'Annotator2', 'Annotator3'], ['Annotator0', 'Annotator4', 'Annotator5', 'Annotator6'],
                                        ['Annotator11', 'Annotator8', 'Annotator9', 'Annotator10'],
                                        ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                        ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                        ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                        ['Annotator4', 'Annotator5', 'Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                        None],
            "Valid Annotators": ["['Annotator1', 'Annotator2', 'Annotator3']", "[ 'Annotator4', 'Annotator5', 'Annotator6']",
                                 "['Annotator8', 'Annotator9', 'Annotator10']",
                                 ['Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],  ['Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                 ['Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'], ['Annotator6', 'Annotator8', 'Annotator9', 'Annotator10'],
                                 None],
            "Entropy": ["-", "-", "-",
                        [[4.0, 4]],
                        [[4.0, 4]],
                        [[4.0, 4]],
                        [[4.0, 4]],
                        None]
        }
        df = pd.DataFrame(value_dict)
        updated_df = main(an_type="ANNOTATOR-UPDATE", test=True, test_df=df)
        self.assertSetEqual(set(updated_df["Admitted Annotators"].iloc[8]),
                            set(['Annotator6', 'Annotator8', 'Annotator9', 'Annotator10']))