"""
    You don't need this

    legacy file for applying functions on a different convo dataset
"""
from typing import List, Dict
from convokit import Corpus, download
from collections import defaultdict



class CorpusWrapper(Corpus):
    """
        overwrite Corpus class from convokit for easier direct application on specific use case
    """

    def __init__(self, convokit_key: str = "subreddit-Cornell"):
        Corpus.__init__(self, filename=download(convokit_key))

    def get_2p_convos(self):
        new_corpus = self.filter_conversations_by(lambda conv: len(conv.get_speaker_ids()) == 2)
        new_corpus.update_speakers_data()
        return new_corpus

    def get_random_convo_id(self):
        return self.random_conversation().id

    def print_convo(self, convo_id: str):
        convo = self.get_conversation(convo_id)
        convo.print_conversation_structure(lambda utt: f"{utt.speaker.id}: {utt.text}")


def directed_utt_pairs(corpus_object: Corpus) -> Dict[str, List[str]]:
    """
    get utterance-reply pairs

    :param corpus_object:
    :return:

    """
    pairs = defaultdict(list)
    for u2 in corpus_object.iter_utterances():
        if u2.speaker is not None and u2.reply_to is not None \
                and u2.reply_to in corpus_object.utterances.keys() and len(u2.text) > 0:
            u1 = corpus_object.get_utterance(u2.reply_to)

            if u1.speaker is not None and len(u1.text) > 0:
                pairs[u1.id].append(u2.id)
    return pairs




