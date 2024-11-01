"""
    Wrapper for MediaSum dataset
        Interview Github Repo: https://github.com/MEDIA-DIALOG/interview-media-analysis
"""
import os
from typing import List, Dict
import json
import pandas as pd
import random
import ast

from paraphrase.utility.annotation_df import get_unique_interview_ids
from paraphrase.utility.project_functions import get_dir_to_src
from paraphrase.set_id_consts import get_dir_to_ms
from paraphrase.utility.qualtrics_survey import tokenize_for_highlight_choices
from paraphrase.lead_pc import get_author_annotations

MEDIASUM_PATH = get_dir_to_ms()
CUT_MEDIASUM_PATH = get_dir_to_src() + "/../result/Data/cut_news_dialogue.tsv"
"""
    stems from 2021 Zhu publication: https://aclanthology.org/2021.naacl-main.474/
    downloaded from: https://drive.google.com/file/d/1ZAKZM1cGhEw2A4_n4bGGMYyF8iPjLZni/view?usp=sharing
        as described at https://github.com/zcgzcgzcg1/MediaSum/tree/main/data
    size: 4.5 GB
    example json element:
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
"""
MEDIASUM_SPLIT_PATH = "../../data/MediaSum/train_val_test_split.json"
"""
    {"train": ["NPR-1", "NPR-2", ..., "NPR-49502", "CNN-1", "CNN-2", ..., "CNN-414241"],
     "val": ["CNN-275527", "CNN-275433", ..., "NPR-48143", ...],
     "test": ["NPR-12", "NPR-134", ..., "CNN-86", ...]
     }
"""
RANDOM_STATE = 42


def df_from_csv(csv_path):
    return pd.read_csv(csv_path, sep="\t")


class Interview:
    SUMMARY = "summary"


class TripleIDs:
    """
        the column names as used for the triples saved in CSV files
    """
    CONVO_ID = "Interview ID"
    IE_UTTERANCE_ID = "First IE Utterance ID"

    IE_IDS = "Consecutive List of IE IDs"
    IR_IDS = "Consecutive List of IR IDs"
    IE2_IDS = "Consecutive List of IE reply IDs to IR"

    def __init__(self, tsv_path: str):
        df_from_csv(tsv_path)

    @staticmethod
    def get_unique_id(interview_id: str, index_first_IE: int):
        return f"{interview_id}-{index_first_IE}"


class MediaSumProcessor:
    """
        Wrapper around MediaSum dataset downloaded via https://github.com/zcgzcgzcg1/MediaSum/tree/main/data
        that also processes the data as needed, mainly:
            + filter interviews (2 person & max tokens)
            + split interviews in TRAIN/DEV/TEST

        main purpose is to save the IDs for use as training data/annotation data
    """
    CONVO_ID = "Interview ID"
    IE_UTTERANCE_ID = "First IE Utterance ID"

    IE_IDS = "Consecutive List of IE IDs"
    IR_IDS = "Consecutive List of IR IDs"
    IE2_IDS = "Consecutive List of IE reply IDs to IR"

    IE_REPLY_UTTERANCE = "IE Reply Utterance"
    IR_UTTERANCE = "IR Utterance"
    IE_UTTERANCE = "IE Utterance"

    def __init__(self, interview_path=MEDIASUM_PATH, uncut=False):
        # MediaSum Data
        if (interview_path == MEDIASUM_PATH) and (not os.path.exists(MEDIASUM_PATH)):
            # Split the path and get the second part onwards
            _, remaining_path = interview_path.split('/', 1)
            interview_path = remaining_path
        self.interview_json_path = interview_path
        self.interviews = None  # was dataframe loaded?
        # self.split_json = split_path
        # self.split = None

        self.twop = False
        self.filtered_ie = [0, 0]  # number of filtered IE utterances because of few # tokens
        self.filtered_ir = [0, 0]  # number of filtered IR utterances because of few # tokens
        self.filtered_address = [0, 0]  # number of filtered interviews because of address phrases
        self.host_guest_confusion_ctr = [0, 0]  # number of filtered interviews because of switched guest/hose position
        self.ctr_not_2 = 0
        self.uncut = uncut

    def load_interview_data(self, recreate=False):
        """
            load interview data into json
            set recreate to True to load the complete corpus not just the corpus that was considered in the project
        :return:
        """

        if (not os.path.exists(CUT_MEDIASUM_PATH)) or recreate or self.uncut:
            # news_dialogue.json consists of only one line --> cannot read in line by line
            self.interviews = pd.read_json(self.interview_json_path)

            if not self.uncut:
                # get interview IDs that were annotated
                all_screening, _, _ = get_author_annotations()
                # get all interview IDs for all_screening
                interview_ids = get_unique_interview_ids(all_screening)
                # subselect interviews with interview_ids only
                self.interviews = self.interviews[self.interviews["id"].isin(interview_ids)]
                # save to file
                self.interviews.to_csv(CUT_MEDIASUM_PATH, sep="\t", index=False)

        else:
            self.interviews = pd.read_csv(CUT_MEDIASUM_PATH, sep="\t")
            self.interviews['speaker'] = self.interviews['speaker'].apply(ast.literal_eval)
            self.interviews['utt'] = self.interviews['utt'].apply(ast.literal_eval)
        # print(self.interviews.size) # 3708768 rows
        print(self.interviews.head())

    # extract from dataset with given IDs

    def get_interview_from_id(self, i_id):
        if self.interviews is None:
            self.load_interview_data()
        hits = self.interviews[self.interviews["id"] == i_id]
        if hits.shape[0] != 1:
            raise ValueError("interview id does not exist or occurs more than once")
        else:
            return hits.to_dict(orient="records")[0]

    def get_utterances(self, i_id):
        """
            get the utterances from the interview with the given id
            merge the utterances of the same speaker
        """
        interview = self.get_interview_from_id(i_id)
        speakers = interview["speaker"]
        utts = interview["utt"]
        # merge utterances of the same speaker in a string, return a list
        speaker_utts = []
        cur_speaker = speakers[0]
        cur_utt = utts[0]
        for speaker, utt in zip(speakers[1:], utts[1:]):
            if speaker == cur_speaker:
                cur_utt += " " + utt
            else:
                speaker_utts.append(cur_utt)
                cur_speaker = speaker
                cur_utt = utt
        speaker_utts.append(cur_utt)
        return speaker_utts


    def get_guest_utt_from_id(self, q_id) -> str:
        return self.get_qdict_from_qid(q_id)["guest utterance"]

    def get_host_utt_from_id(self, q_id) -> str:
        return self.get_qdict_from_qid(q_id)["host utterance"]

    def get_summary_from_id(self, q_id) -> str:
        return self.get_qdict_from_qid(q_id)["summary"]

    def get_qdict_from_qid(self, q_id) -> Dict[str, str]:
        """
            TODO: updated to return dict
                -> not updated for all calls yet, should be caught by typing though
        :param q_id: ID of the form  'CNN-67148-13' where 'CNN-67148' is the identifier as used in MediaSum and 13
            is the index of the first guest utterance in the (guest, host)-utterance pair as defined by q_id
        :return: (guest, host)- utterance pair as well as contextual info
        """
        if self.interviews is None:
            self.load_interview_data()

        # _make_names_unique
        interview = self.get_interview_from_id(q_id[:q_id.rfind('-')])
        speakers = self._make_names_unique(interview["speaker"])
        id_start = int(q_id[q_id.rfind('-') + 1:])
        guest_name = speakers[int(q_id[q_id.rfind('-') + 1:])]
        host_name = speakers[0]
        id_guest_end = id_start + speakers[id_start:].index(host_name)
        id_host_end = id_guest_end + speakers[id_guest_end:].index(guest_name)

        ie_utt = self.get_utt(interview, str(list(range(id_start, id_guest_end))))
        ir_utt = self.get_utt(interview, str(list(range(id_guest_end, id_host_end))))
        summary = interview['summary']

        return {
            "host utterance": ir_utt,
            "guest utterance": ie_utt,
            "summary": summary,
            "host name": host_name,
            "guest name": guest_name,
            "date": interview["date"],
            "q_id": q_id
        }

    @staticmethod
    def load_split_data(splitpath: str):
        """
            load train/dev/test split into dict with interview ids
            :param splitpath: path to json file that is to be read in
        :return:
        """
        with open(splitpath) as f:
            return json.load(f)
        # print(self.split.keys())

    def filter_2_person(self):
        """
            only keep interviews with two participants (assumed to be one host and one guest)
        :return:
        """
        if self.interviews is None:
            self.load_interview_data()
        if not self.twop:
            npr_size = self.interviews[self.interviews["id"].str.contains("NPR")].shape[0]
            print(f"Starting to filter for only 2-person interviews from an initial interview dataset of size"
                  f" {self.interviews.shape[0]} with {npr_size} NPR interviews and"
                  f" {self.interviews.shape[0] - npr_size} CNN interviews...")
            self.interviews = self.interviews[self.interviews.apply(lambda x: self._is2p(x), axis=1)]
            self.twop = True
            npr_size = self.interviews[self.interviews["id"].str.contains("NPR")].shape[0]
            print(f"{self.interviews.shape[0]} remain as 2-person dialogs out of which {npr_size} "
                  f"are NPR interviews and {self.interviews.shape[0] - npr_size} are CNN interviews")
            print(f"removed interviews because of guest/host confusion is at {self.host_guest_confusion_ctr}, "
                  f"rest removed because of 2 person condition")

    @staticmethod
    def get_host_and_guest_name(speaker_list):
        unique_s_list = MediaSumProcessor._make_names_unique(speaker_list)
        host_name = unique_s_list[0]
        guest_name = set(unique_s_list)
        guest_name.remove(host_name)
        guest_name = list(guest_name)[0]
        return host_name, guest_name

    @staticmethod
    def _make_names_unique(speaker_list):
        """
        For example
            ['MADELEINE BRAND, host', 'Mr. RANDY HALL', 'STEVE PROFFITT', 'Ms. SUSAN STURGILL', 'PROFFITT']
        becomes
            ['MADELEINE BRAND, host', 'Mr. RANDY HALL', 'STEVE PROFFITT', 'Ms. SUSAN STURGILL', 'STEVE PROFFITT']
        s.t. all speakers have UNIQUE identifiers

        only handles cases where one author identifier is contained completely in the other
            i.e., not S. PROFFITT vs. STEVE PROFFITT are not matched

        :param speaker_list:
        :return: list of same length as speaker list but non-unique identifiers replaced by unique one, keeps correct ordering
        """
        assert type(speaker_list) == list, "speaker_list must be a list"
        ordered_speakers = sorted(speaker_list, key=len)
        name_to_unique = {}
        for i, name in enumerate(ordered_speakers):
            substr_of = [speaker for speaker in ordered_speakers[i + 1:] if name in speaker]
            if len(substr_of) > 0:
                name_to_unique[name] = substr_of[-1]
        return [name_to_unique[speaker] if speaker in name_to_unique else speaker for speaker in speaker_list]

    def _is2p(self, json_convo):
        """
            only called in filter_2_person
            ATTENTION: (ugly but) is used to increase host/guest confusion counter
                --> if it is called somewhere else this will mess with that statistic
        :param json_convo: json of the conversation
        :return: checks if exactly 2 speakers are part of the conversation AND
            not "host" or "anchor" in the first speaker position, or both speakers are designated as hosts/anchors
        """
        try:
            unique_speaker_list = self._make_names_unique(json_convo['speaker'])
            speaker_list = sorted(set(unique_speaker_list),
                                  key=unique_speaker_list.index)  # list(set(json_convo['speaker']))
            ctr_host_confusion = 0
            if len(speaker_list) == 2:
                if ("anchor" in speaker_list[1].lower()) or ("host" in speaker_list[1].lower()):  # host in guest pos
                    if "NPR" in json_convo['id']:
                        self.host_guest_confusion_ctr[0] += 1
                    else:
                        self.host_guest_confusion_ctr[1] += 1
                    rnd_float = random.random()
                    if rnd_float < 0.05:
                        ctr_host_confusion += 1
                        print(f"Example {ctr_host_confusion} of host not in the right place")
                        print(f"\t{speaker_list}")
                    return False
                else:
                    return True
            else:
                rnd_float = random.random()
                if rnd_float < 0.0001:
                    self.ctr_not_2 += 1
                    print(f"Example {self.ctr_not_2} of not 2 speakers:")
                    print(f"\t{set(unique_speaker_list)}")
                return False
        except:
            raise ValueError("Something with the given argument json_convo went wrong."
                             " It either is not a json object or it does not have the attribute speaker.")

    def print_convo(self, json_convo):
        print(f"{json_convo['id']}: {json_convo['program']} on {json_convo['date']}"
              f" with the title {json_convo['title']}, src-url:{json_convo['url']}")
        for speak, utt in zip(json_convo["speaker"], json_convo["utt"]):
            print(f"\t \033[1m{speak}\033[0m: {utt}")

    def sample_convos(self, nbr: int, min_len=False):
        """
            sample nbr conversations with length of at least min_len
            helper function
        :param min_len:
        :param nbr:
        :return: dataframe with nbr sampled rows from self.interviews dataframe
        """
        if self.interviews is None:
            self.load_interview_data()
        if not min_len:
            df = self.interviews
        else:
            df = self.interviews[self.interviews['utt'].map(len) > min_len]  # .apply(lambda x: self.is2p(x), axis=1)
            print(f"{df.shape[0]} interviews fulfill the minlen {min_len} condition out of which "
                  f"{df[df['id'].str.contains('NPR')].shape[0]} are NPR interviews")
        return df.sample(n=nbr, random_state=RANDOM_STATE)

    @staticmethod
    def _get_start_pos(index_list: List[int]):
        """
            get the starting numbers of consecutive sequences
                [4, 6, 8, 10, 11, 12, 14] --> [4, 6, 8, 10, 14]
        :param index_list:
        :return:
        """
        prev_elem = index_list[0]
        new_list = [prev_elem]
        for elem in index_list[1:]:
            if prev_elem + 1 != elem:
                new_list.append(elem)
            prev_elem = elem
        return new_list

    def _iter_filtered_instances(self):
        """
        generator over consecutive interviewee (ie) interviewer (ir) utterance pairs,
        that fulfil the preprocessing conditions, currently this means:
          - continued by at least one more ie utterance, i.e., an "outcome"
          - more than 2 tokens for both IE and IR

        ASSUMING a 2-person conversation (filter_2_person() should be called first)
            & that the person starting off the conversation is the host
        DOES CURRENTLY NOT do any tokenization, stemming etc. or consider length limits

        UPDATES the interviews dataframe to only include interviews that were not filtered out completely

        :return: yields dictionaries of the form
            {"id": "NPR-1",
             "ie_utt_pos": [3],  # indices for the (continuous) IE utterances in the utterance list
             "ir_utt_pos": [4],  # indices for the (continuous) IR utterance in the utterance list
             "nxt_ie_utt_pos": [5]  # indices for the (continuous) IE reply in the utterance list
             "ie_utt": [""],   # the string of the actual interviewee utterance
             "ir_utt": [""],   # the string of the actual interviewer utterance that follows the ie utt
             "nxt_ie_utt": [""]  # the string of the actual IE reply utterance that follows the IR utt
            }

        """
        # only makes sense for 2-person interviews
        self.filter_2_person()

        leftover_i_ids = set()

        for _, interview in self.interviews.iterrows():
            i_id = interview["id"]

            # DEBUG
            # if i_id == "NPR-31754":
            #     print("Tada")

            i_s_list = self._make_names_unique(interview["speaker"])
            i_u_list = interview["utt"]
            nbr_utts = len(i_s_list)

            # ASSUMPTION: first speaker is interviewer
            host_name = i_s_list[0]

            # get position of host and guest utterances
            guest_pos = [i for i, speaker in enumerate(i_s_list) if speaker != host_name]
            host_pos = [i for i, speaker in enumerate(i_s_list) if speaker == host_name]
            start_indices_gp = self._get_start_pos(guest_pos)
            start_indices_hp = self._get_start_pos(host_pos)

            # test if interview is long enough to contain any pairs: 5 host positions =^ at least 4 (guest, host)-pairs
            # if len(start_indices_gp) <= 3:  # 1 (welcome) + 2 (goodbye) = 3 that are filtered out
            if len(start_indices_hp) <= 4:  # first pair (welcome pair =^ 2 hosts) + 2 last pairs (goodbye) = 4 that are filtered out
                if "NPR" in i_id:
                    self.filtered_address[0] += 1
                else:
                    self.filtered_address[1] += 1
                continue

            # remove the first turn (i.e., all consecutive first utterances) by guest
            start_indices_gp = start_indices_gp[1:]
            guest_pos = guest_pos[guest_pos.index(start_indices_gp[0]):]
            host_pos = [i for i in range(guest_pos[0], nbr_utts) if i not in guest_pos]
            start_indices_hp = self._get_start_pos(host_pos)

            cur_guest = 0
            while cur_guest < len(start_indices_gp) - 2:  # up until 2nd to last pair
                ie_utts = list(range(start_indices_gp[cur_guest], start_indices_hp[cur_guest]))
                ir_utts = list(range(start_indices_hp[cur_guest], start_indices_gp[cur_guest + 1]))
                if len(ie_utts) > 1 or len(i_u_list[ie_utts[0]].split(" ")) > 2 and \
                        (len(ir_utts) > 1 or len(i_u_list[ir_utts[0]].split(" ")) > 2):
                    # only keep ie and ir utt with more than 3 tokens
                    nxt_utts = list(range(start_indices_gp[cur_guest + 1], start_indices_hp[cur_guest + 1]))
                    leftover_i_ids.add(i_id)
                    yield {"conv_id": i_id,
                           "utt_id": guest_pos[cur_guest],
                           "url": interview["url"],
                           "ie_utt_pos": ie_utts,
                           "ie_utt": [i_u_list[i] for i in ie_utts],
                           "ir_utt_pos": ir_utts,
                           "ir_utt": [i_u_list[i] for i in ir_utts],
                           "nxt_ie_utt_pos": nxt_utts,
                           "nxt_ie_utt": [i_u_list[i] for i in nxt_utts]
                           }
                else:
                    if len(ie_utts) == 1 and len(i_u_list[ie_utts[0]].split(" ")) <= 2:
                        if "NPR" in i_id:
                            self.filtered_ie[0] += 1
                        else:
                            self.filtered_ie[1] += 1
                    else:
                        if "NPR" in i_id:
                            self.filtered_ir[0] += 1
                        else:
                            self.filtered_ir[1] += 1
                    # have a look at what is excluded & print 1% of the time
                    if random.random() < 0.003:
                        print(f"{i_id}-{guest_pos[cur_guest]}")
                        print([i_u_list[i] for i in ie_utts])
                        ir_utts = list(range(start_indices_hp[cur_guest], start_indices_gp[cur_guest + 1]))
                        print([i_u_list[i] for i in ir_utts])
                        print("----")
                cur_guest += 1

        # only keep interviews that had at least one triple
        self.interviews = self.interviews[self.interviews["id"].isin(leftover_i_ids)]

    def filter_ieirpairs_to_df(self):
        """
            iterates over (ie1, ir, ie2) 2-person interview triples, filters them acc. to _iter_filtered_instances and
                saves the ids for later extraction in df format
        :return:
        """
        # For each (ie1, ir, ie2) triple save
        conv_id = []  # interview id
        ie_start_id = []  # id of the first ie utterance
        ie_ids = []
        ir_ids = []
        ie2_ids = []
        ie_utts = []  # actual ie utt
        ir_utts = []  # actual ir utt
        ie_rply = []  # actual ie reply

        # only keep interviews with 2 people, where host/anchor is NOT part of the designated guest identifier
        self.filter_2_person()

        t_c_ids = self.interviews["id"].unique()
        print(f"There are {len(set(t_c_ids))} unique 2-person interviews")
        print(f"    \t out of which {len(set([c_id for c_id in t_c_ids if 'CNN' in c_id]))} cnn and "
              f"{len(set([c_id for c_id in t_c_ids if 'NPR' in c_id]))} npr interviews")
        # print(f"    \t out of which {len([c_id for c_id in t_c_ids if 'CNN' in c_id])} cnn triples and "
        #       f"{len([c_id for c_id in t_c_ids if 'NPR' in c_id])} npr triples interviews")

        for pair_obj in self._iter_filtered_instances():
            conv_id.append(pair_obj["conv_id"])
            ie_ids.append(pair_obj["ie_utt_pos"])
            ir_ids.append(pair_obj["ir_utt_pos"])
            ie2_ids.append(pair_obj["nxt_ie_utt_pos"])

            ie_start_id.append(pair_obj["ie_utt_pos"][0])
            ie_utts.append("\n".join(pair_obj["ie_utt"]))
            ir_utts.append("\n".join(pair_obj["ir_utt"]))
            ie_rply.append("\n".join(pair_obj["nxt_ie_utt"]))

        print(f"{len(set(conv_id))} unique 2-person interviews were kept after filtering")
        print(f"    \t out of which {len(set([c_id for c_id in conv_id if 'CNN' in c_id]))} cnn and "
              f"{len(set([c_id for c_id in conv_id if 'NPR' in c_id]))} npr interviews")
        print(f"    \t out of which {len([c_id for c_id in conv_id if 'CNN' in c_id])} cnn triples and "
              f"{len([c_id for c_id in conv_id if 'NPR' in c_id])} npr triples interviews")

        print(f"    \t {self.filtered_ie} pairs were removed due to short IE utterances")
        print(f"    \t and {self.filtered_ir} pairs were removed due to short IR utterances")
        print(f"    \t and {self.filtered_address} interviews were removed due to to few pairs (less than 4)")

        columns = [self.CONVO_ID, self.IE_UTTERANCE_ID, self.IE_UTTERANCE, self.IR_UTTERANCE, self.IE_REPLY_UTTERANCE]
        df = pd.DataFrame(list(zip(conv_id, ie_start_id, ie_utts, ir_utts, ie_rply)), columns=columns)

        columns = [self.CONVO_ID, self.IE_UTTERANCE_ID, self.IE_IDS, self.IR_IDS, self.IE2_IDS]
        df_ids = pd.DataFrame(list(zip(conv_id, ie_start_id, ie_ids, ir_ids, ie2_ids)), columns=columns)
        return df, df_ids

    def ieir_ids_to_csv(self, path):
        _, df_ids = self.filter_ieirpairs_to_df()
        df_ids.to_csv(path, sep="\t")
        return df_ids

    def ieir_from_id_csv(self, ieir_path, to_save):
        """
            -- OBSOLETE --
            get dataset from csv with ids (e.g., generated from ieir_ids_to_csv)
        :param ieir_path:
        :param to_save:
        :return:
        """
        df_ids = pd.read_csv(ieir_path, sep="\t")

        conv_id = []
        ie_start_id = []
        ie_utts = []
        ir_utts = []
        ie_rply = []

        for index, row in df_ids.iterrows():
            convo = self.get_interview_from_id(row[self.CONVO_ID])
            conv_id.append(row[self.CONVO_ID])
            ie_start_id.append(row[self.IE_UTTERANCE_ID])
            ie_utts.append(self.get_utt(convo, row[self.IE_IDS]))
            ir_utts.append(self.get_utt(convo, row[self.IR_IDS]))
            ie_rply.append(self.get_utt(convo, row[self.IE2_IDS]))

        columns = [self.CONVO_ID, self.IE_UTTERANCE_ID, self.IE_UTTERANCE, self.IR_UTTERANCE, self.IE_REPLY_UTTERANCE]
        df = pd.DataFrame(list(zip(conv_id, ie_start_id, ie_utts, ir_utts, ie_rply)), columns=columns)
        df.to_csv(to_save, sep="\t")

        return df

    @staticmethod
    def get_utt(interview_obj: dict, utt_ids: str):
        """
        function that gets the interview utterances from the given ids and returns as combined string
        CHANGED: added ws before "\n"

        :param interview_obj: object of an interview, e.g., when get_interview_from_id is called
        :param utt_ids: string of a list of integers that signify which utterances should be extracted
        :return: string with new lines where different turns are combined
        """
        return " \n".join([interview_obj["utt"][i] for i in ast.literal_eval(utt_ids)])

    def filter_and_shuffle_ghpairs_to_tsv(self, ghID_path: str, seed: int = RANDOM_STATE):
        """
            filter out non 2-person interviews and those of uninteresting format
            split data into train dev test and save in according files
        :param ghID_path:
        :param seed:
        :return:
        """
        # filter conversations triples according to filtering criteria (includes 2-person filter)
        _, gh_pairs_ids = self.filter_ieirpairs_to_df()

        # get all interview ids
        id_list = list(gh_pairs_ids[self.CONVO_ID].unique())

        # shuffle them according to seed
        random.seed(seed)
        random.shuffle(id_list)

        # change dataframe according to shuffling of interview order
        gh_pairs_ids = gh_pairs_ids.set_index(TripleIDs.CONVO_ID).loc[id_list].reset_index()
        print(
            f"Filtered and shuffled interview guest, host items with DISTINCT {len(id_list)} interviews with \n "
            f"\t {gh_pairs_ids.shape[0]} (IE, IR)-pairs.")

        gh_pairs_ids.to_csv(ghID_path, sep="\t")

    def _train_dev_test_listsplit(self, id_list, seed=RANDOM_STATE, use_mediasum_split=False):
        """
            either use a given the train/dev/test split from json file to get the interviews for the respective lists
            or randomly separate 70-15-15
        :param id_list:
        :return:
        """
        # if use_mediasum_split:  # OBSOLETE
        #     self.load_split_data()
        #     train_ids = list(set(id_list).intersection(set(self.split["train"])))
        #     dev_ids = list(set(id_list).intersection(self.split["val"]))
        #     test_ids = list(set(id_list).intersection(self.split["test"]))
        # else:
        random.seed(seed)
        train_ids = random.sample(id_list, k=round(len(id_list) * 0.7))
        dev_ids = random.sample(list(set(id_list) - set(train_ids)),
                                k=round(len(set(id_list) - set(train_ids)) * 0.5))
        test_ids = list(set(id_list) - set(dev_ids) - set(train_ids))

        assert (not set(train_ids) & set(dev_ids))
        assert (not set(train_ids) & set(test_ids))
        assert (not set(dev_ids) & set(test_ids))

        random.shuffle(train_ids)
        random.shuffle(dev_ids)
        random.shuffle(test_ids)

        return train_ids, dev_ids, test_ids


    def get_guest_hl_from_id(self, q_id, hl_ids) -> str:
        guest_utt = self.get_guest_utt_from_id(q_id)
        return self._get_utt_hl(guest_utt, hl_ids)

    def get_host_hl_from_id(self, q_id, hl_ids) -> str:
        host_utt = self.get_host_utt_from_id(q_id)
        return self._get_utt_hl(host_utt, hl_ids)

    def _get_utt_hl(self, utt, hl_ids):
        if type(hl_ids) == str and hl_ids != "-":
            hl_ids = ast.literal_eval(hl_ids)
        elif hl_ids == "-":
            hl_ids = []
        utt_tokens = tokenize_for_highlight_choices(utt)
        return " ".join(word.upper() if i + 1 in hl_ids else word for i, word in enumerate(utt_tokens))

    @staticmethod
    def ieir_from_csv(csv_path):
        df = df_from_csv(csv_path)
        for index, row in df.iterrows():
            yield row.to_dict()

    def iter_sample_pairs(self, n: int, csv_path: str):
        """
            sample n (ie, ir) pairs and yield them iteratively
        :param n: integer, the number of (ie, ir) pairs that are sampled
        :param csv_path: path to the data to sample from
        :return: iterator over question_id, ie_utt, ir_utt, summary
        """
        s_df = df_from_csv(csv_path)
        s_df = s_df.sample(n)

        for _, row in s_df.iterrows():
            i_id = row[MediaSumProcessor.CONVO_ID]
            interview_dict = self.get_interview_from_id(i_id)
            summary = interview_dict[Interview.SUMMARY]
            ie_utt = self.get_utt(interview_dict, row[MediaSumProcessor.IE_IDS])
            ir_utt = self.get_utt(interview_dict, row[MediaSumProcessor.IR_IDS])
            q_id = f"{i_id}-{row[MediaSumProcessor.IE_UTTERANCE_ID]}"
            yield q_id, ie_utt, ir_utt, summary

    def iter_for_annotation(self, ieir_tsv: str):
        """
            gets utterances/summaries/names etc. from mediasum corpus
                & removes pairs with too many (more than 200) tokens

        :param ieir_tsv:
        :return:
        """

        # split = self.load_split_data(shuffled_split_json)
        # shuffled_interview_ids = split["train"]

        i_df = pd.read_csv(ieir_tsv, sep='\t')
        shuffled_interview_ids = i_df[
            TripleIDs.CONVO_ID].unique()  # unique keeps order (https://pandas.pydata.org/docs/reference/api/pandas.unique.html)
        max_per_interview = 5
        removed_pairs = 0
        print(f"Sampling a maximum of {max_per_interview} pairs per interview "
              f"with a limit of 200 tokens per utterance ...")

        for i_id in shuffled_interview_ids:
            interview_dict = self.get_interview_from_id(i_id)
            if ("anchor" in self.get_host_and_guest_name(interview_dict['speaker'])[1].lower()) \
                    or ("host" in self.get_host_and_guest_name(interview_dict['speaker'])[1].lower()):
                # remove interview cases where anchor/host slips through as guest...
                print(" this should never happen: wrongly assigned anchor/hosts should have been removed")
                continue
            summary = interview_dict[Interview.SUMMARY]

            nbr_pairs_interview = 0

            for _, trip_row in i_df.loc[i_df[TripleIDs.CONVO_ID] == i_id].iterrows():
                if nbr_pairs_interview < max_per_interview:
                    ie_utt = self.get_utt(interview_dict, trip_row[TripleIDs.IE_IDS])
                    ir_utt = self.get_utt(interview_dict, trip_row[TripleIDs.IR_IDS])
                    try:
                        # test if can call highlight choices (i.e., at most 200 ws)
                        tokenize_for_highlight_choices(ie_utt)
                        tokenize_for_highlight_choices(ir_utt)

                    except ValueError:
                        # text too long to tokenize
                        removed_pairs += 1
                        print(f"removed {removed_pairs} pairs so far due to too many tokens ...")
                        continue
                    q_id = TripleIDs.get_unique_id(interview_id=i_id,
                                                   index_first_IE=trip_row[TripleIDs.IE_UTTERANCE_ID])
                    yield q_id, ie_utt, ir_utt, summary
                    nbr_pairs_interview += 1
                else:
                    break

    def iter_all_qs(self, ieir_tsv: str):
        """
            get all question ids from the given tsv
        :param ieir_tsv:
        :return:
        """
        i_df = pd.read_csv(ieir_tsv, sep='\t')
        interview_ids = i_df[TripleIDs.CONVO_ID].unique()

        for i_id in interview_ids:
            interview_dict = self.get_interview_from_id(i_id)
            if ("anchor" in self.get_host_and_guest_name(interview_dict['speaker'])[1].lower()) \
                    or ("host" in self.get_host_and_guest_name(interview_dict['speaker'])[1].lower()):
                # remove interview cases where anchor/host slips through as guest...
                print(" this should never happen: wrongly assigned anchor/hosts should have been removed")
                continue

            summary = interview_dict[Interview.SUMMARY]

            for _, trip_row in i_df.loc[i_df[TripleIDs.CONVO_ID] == i_id].iterrows():
                ie_utt = self.get_utt(interview_dict, trip_row[TripleIDs.IE_IDS])
                ir_utt = self.get_utt(interview_dict, trip_row[TripleIDs.IR_IDS])

                q_id = TripleIDs.get_unique_id(interview_id=i_id,
                                               index_first_IE=trip_row[TripleIDs.IE_UTTERANCE_ID])
                yield q_id, ie_utt, ir_utt, summary
