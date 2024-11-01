"""
    You don't need this

    legacy test file
"""
from unittest import TestCase
from paraphrase.utility.convokit_helper import CorpusWrapper


class TestRelationship(TestCase):
    def setUp(self) -> None:
        self.corpus = CorpusWrapper("subreddit-relationship_advice")
        self.corpus.print_summary_stats()

    def test_2p_convo(self):
        two_p_corpus = self.corpus.get_2p_convos()
        two_p_corpus.print_summary_stats()
        # print a random convo of length >= 4
        rand_convo = two_p_corpus.random_conversation()
        while len(rand_convo.get_utterance_ids()) < 4:
            rand_convo = two_p_corpus.random_conversation()
        two_p_corpus.print_convo(rand_convo.id)

        # make sure the found conversations actually have ony 2 participants
        for convo in two_p_corpus.iter_conversations():
            self.assertEqual(len(convo.get_speaker_ids()), 2)


class TestAwry(TestCase):
    def setUp(self):
        self.corpus = CorpusWrapper("conversations-gone-awry-corpus")
        self.corpus.print_summary_stats()

    def test_get_awry_convo(self):
        self.corpus.print_convo(self.corpus.get_random_convo_id())

    def test_2p_convo(self):
        two_p_corpus = self.corpus.get_2p_convos()
        two_p_corpus.print_summary_stats()
        rand_conv = two_p_corpus.random_conversation()
        while rand_conv.meta['conversation_has_personal_attack']:
            rand_conv = two_p_corpus.random_conversation()
        two_p_corpus.print_convo(rand_conv.id)
        for convo in two_p_corpus.iter_conversations():
            self.assertEqual(len(convo.get_speaker_ids()), 2)
