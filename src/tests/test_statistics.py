"""
    testing agreement scores etc.
"""
from unittest import TestCase
from paraphrase.utility.stats import (transformer_scores, cohen_kappa_agreement, jaccard_overlap_for_highlights,
                                      calc_krippendorff_alpha)
import numpy as np

class Test(TestCase):
    def test_calculate_agreement(self):
        labeler1 = [1, 0, 1, 1, 0, 1]
        labeler2 = [0, 0, 1, 1, 0, 0]
        print(cohen_kappa_agreement(labeler1, labeler2))

    def test_bertscore(self):
        h1 = "highlight words by boldening"
        h2 = "highlight words by boldening them"
        h3 = "you highlight words"
        b_score = transformer_scores([[h1, h2, h3]])[0]
        b_1 = transformer_scores([[h1, h2]])[0]
        b_2 = transformer_scores([[h2, h3]])[0]
        b_3 = transformer_scores([[h1, h3]])[0]
        print(b_score)
        self.assertLess(b_score, b_1)
        self.assertGreater(b_score, b_2)
        self.assertGreater(b_score, b_3)
        self.assertGreater(b_score, 0.5)

    def test_overlap(self):
        h1 = [1, 5, 6]
        h2 = [1, 5]
        h3 = [1, 6]
        overlap = jaccard_overlap_for_highlights([[h1, h2]])
        self.assertEqual(2/3, overlap)

    def test_understand_bscore(self):
        # And if the TARDIGRADES did survive this crash landing, are they still SANDWICHED IN THIS KIND OF DVD TYPE THING?
        # And if the tardigrades did survive this crash landing, are they still SANDWICHED IN THIS KIND OF DVD TYPE THING?
        h1 = "tardigrades sandwiched in this kind of DVD type thing?"
        h2 = "sandwiched in this kind of DVD type thing?"
        print(transformer_scores([[h1, h2]])[0])

        h1 = "tardigrades are they still sandwiched in this kind of DVD type thing?"
        h2 = "sandwiched in this kind of DVD type thing?"
        print(transformer_scores([[h1, h2]])[0])

        # So these stories have been recounted in media reports, but are these people talking publicly about the threats? Has it changed their behavior at all? I mean, clearly if ONE OF THEM LEFT THE COUNTRY, it is to some degree.
        # SO THESE STORIES have been recounted in media reports, but are these people talking publicly about the threats? Has it changed their behavior at all? I mean, clearly if one of them left the country, it is to some degree.
        h1 = "So these stories"
        h2 = "these stories"
        print(transformer_scores([[h1, h2]])[0])

        h1 = "So these stories"
        h2 = "one of them left the country"
        print(transformer_scores([[h1, h2]])[0])

    def test_calc_krippendorff_alpha(self):
        # for 3 annototators annotating one text with 5 tokens
        annotator_matrix = np.array([[0, 0, 1, 1, 1],
                                     [np.nan, np.nan, np.nan, np.nan, np.nan],
                                     [0, 1, 1, 1, 0]])
        print(calc_krippendorff_alpha(annotator_matrix))
