"""
    test interface with MediaSum Dataset
"""
from unittest import TestCase

from preprocess_interviews import IDS_CSV
from paraphrase.interview_data import MediaSumProcessor
import json


class TestMediaSumLoader(TestCase):

    def setUp(self) -> None:
        self.dummy_interview_path = "paraphrase/utility/fixtures/data/test_interview-corpus.json"
        self.interview = MediaSumProcessor()
        self.split = "../../data/MediaSum/train_val_test_split.json"
        self.dummy_interview = MediaSumProcessor(self.dummy_interview_path)
        self.convo_3p = '{ "id": "NPR-11", "program": "Day to Day","date": "2008-06-10", "url": "https://www.npr.org/templates/story/story.php?storyId=91356794","title": "Researchers Find Discriminating Plants","summary": "The \'sea rocket\' shows preferential treatment to plants that are its kin. Evolutionary plant ecologist Susan Dudley of McMaster University in Ontario discusses her discovery.", "utt": ["This is Day to Day.  I\'m Madeleine Brand.", "And I\'m Alex Cohen.", "Coming up, the question of who wrote a famous religious poem turns into a very unchristian battle.", "First, remember the 1970s? People talked to their houseplants, played them classical music. They were convinced plants were sensuous beings and there was that 1979 movie, \'The Secret Life of Plants.\'", "OK. Thank you.", "That\'s Susan Dudley. She\'s an associate professor of biology at McMaster University in Hamilt on Ontario. She discovered that there is a social life of plants."], "speaker": ["MADELEINE BRAND, host","ALEX COHEN, host", "ALEX COHEN, host","MADELEINE BRAND, host", "Professor SUSAN DUDLEY (Biology, McMaster University)", "MADELEINE BRAND, host"] }'
        self.convo_2p = '{ "id": "NPR-11", "program": "Day to Day","date": "2008-06-10", "url": "https://www.npr.org/templates/story/story.php?storyId=91356794","title": "Researchers Find Discriminating Plants","summary": "The \'sea rocket\' shows preferential treatment to plants that are its kin. Evolutionary plant ecologist Susan Dudley of McMaster University in Ontario discusses her discovery.", "utt": ["This is Day to Day.  I\'m Madeleine Brand.", "And I\'m Alex Cohen.", "Coming up, the question of who wrote a famous religious poem turns into a very unchristian battle.", "First, remember the 1970s? People talked to their houseplants, played them classical music. They were convinced plants were sensuous beings and there was that 1979 movie, \'The Secret Life of Plants.\'"], "speaker": ["MADELEINE BRAND, host","ALEX COHEN", "ALEX COHEN","MADELEINE BRAND, host"] }'
        self.convo_host_guest = '{ "id": "NPR-11", "program": "Day to Day","date": "2008-06-10", "url": "https://www.npr.org/templates/story/story.php?storyId=91356794","title": "Researchers Find Discriminating Plants","summary": "The \'sea rocket\' shows preferential treatment to plants that are its kin. Evolutionary plant ecologist Susan Dudley of McMaster University in Ontario discusses her discovery.", "utt": ["This is Day to Day.  I\'m Madeleine Brand.", "And I\'m Alex Cohen.", "Coming up, the question of who wrote a famous religious poem turns into a very unchristian battle.", "First, remember the 1970s? People talked to their houseplants, played them classical music. They were convinced plants were sensuous beings and there was that 1979 movie, \'The Secret Life of Plants.\'"], "speaker": ["ALEX COHEN", "MADELEINE BRAND, host", "ALEX COHEN", "MADELEINE BRAND, host"] }'
        self.convo_anchor_guest = '{ "id": "NPR-11", "program": "Day to Day","date": "2008-06-10", "url": "https://www.npr.org/templates/story/story.php?storyId=91356794","title": "Researchers Find Discriminating Plants","summary": "The \'sea rocket\' shows preferential treatment to plants that are its kin. Evolutionary plant ecologist Susan Dudley of McMaster University in Ontario discusses her discovery.", "utt": ["This is Day to Day.  I\'m Madeleine Brand.", "And I\'m Alex Cohen.", "Coming up, the question of who wrote a famous religious poem turns into a very unchristian battle.", "First, remember the 1970s? People talked to their houseplants, played them classical music. They were convinced plants were sensuous beings and there was that 1979 movie, \'The Secret Life of Plants.\'"], "speaker": ["ALEX COHEN", "MADELEINE BRAND, anchor", "MADELEINE BRAND, anchor", "ALEX COHEN"] }'

    " ------------ DUMMY TESTING -------------------- "

    def test_load_dummy(self):
        """
            test loading the dummy interview corpus with three interviews,
                but only 1 that will not be filtered out
        :return:
        """
        self.dummy_interview.load_interview_data()
        self.assertListEqual(list(self.dummy_interview.interviews.columns),
                             ['id', 'program', 'date', 'url', 'title', 'summary',
                              'utt', 'speaker'])
        self.assertEqual(self.dummy_interview.interviews.shape[0], 3)
        self.assertEqual(len(self.dummy_interview.interviews["id"].unique()), 3)
        print(self.dummy_interview.interviews)

    def test_2filter_dummy(self):
        """
            1 filtered out  because of 3 persons being present
        :return:
        """
        self.dummy_interview.load_interview_data()
        self.dummy_interview.filter_2_person()
        self.assertEqual(self.dummy_interview.interviews.shape[0], 2)
        self.assertEqual(len(self.dummy_interview.interviews["id"].unique()), 2)

    def test_token_filter_dummy(self):
        """
            1 filtered out  because of 3 persons being present
        :return:
        """
        self.dummy_interview.load_interview_data()
        self.dummy_interview.filter_ieirpairs_to_df()
        self.assertEqual(self.dummy_interview.interviews.shape[0], 1)
        self.assertEqual(len(self.dummy_interview.interviews["id"].unique()), 1)

    def test_is2person(self):
        convo_3p = json.loads(self.convo_3p)
        self.assertFalse(self.interview._is2p(convo_3p))
        convo_2p = json.loads(self.convo_2p)
        self.assertTrue(self.interview._is2p(convo_2p))
        convo_host_guest = json.loads(self.convo_host_guest)
        self.assertFalse(self.interview._is2p(convo_host_guest))
        convo_anchor_guest = json.loads(self.convo_anchor_guest)
        self.assertFalse(self.interview._is2p(convo_anchor_guest))


    def test_filter_double_names(self):
        speaker_list = ['MADELEINE BRAND, host', 'Mr. RANDY HALL', 'STEVE PROFFITT', 'Ms. SUSAN STURGILL', 'PROFFITT',
                        'Ms. BARBARA LEBEY']
        unique_speakers = ['MADELEINE BRAND, host', 'Mr. RANDY HALL', 'STEVE PROFFITT', 'Ms. SUSAN STURGILL',
                           'STEVE PROFFITT', 'Ms. BARBARA LEBEY']
        self.assertSetEqual(set(unique_speakers), set(self.interview._make_names_unique(speaker_list)))
        self.assertListEqual(self.interview._make_names_unique(speaker_list), unique_speakers)

    def test_get_start_pos(self):
        index_list = [4, 6, 8, 10, 11, 12, 14]
        self.assertEqual([4, 6, 8, 10, 14], self.interview._get_start_pos(index_list))

    def test_dummy_ieir_pairs(self):
        """
            check that the dummy json interview is correctly represented in ieir pairs
        :return:
        """
        self.dummy_interview.load_interview_data()
        pair_gen = self.dummy_interview._iter_filtered_instances()
        nbr_pairs = 0
        # start position utterance interviewee
        ie_pos = [4, 6, 8, 10, 13, 15]  # without removing fist & last 2 pair & Mhm-mh: [2, 4, 6, 8, 10, 13, 15, 18, 20]
        # start position utterance interviewer
        ir_pos = [5, 7, 9, 12, 14, 17]  # without removing fist & last 2 pair & Mhm-mh: [3, 5, 7, 9, 12, 14, 17, 19, 21]
        nxt_pos = [6, 8, 10, 13, 15, 18]
        for pair_obj, gt_ie_pos, gt_ir_pos, gt_nxt_pos in zip(pair_gen, ie_pos, ir_pos, nxt_pos):
            nbr_pairs += 1
            conv_id = pair_obj["conv_id"]
            self.assertEqual(conv_id, "NPR-4")
            ie_utt = pair_obj["ie_utt"]
            ir_utt = pair_obj["ir_utt"]
            nxt_ie = pair_obj["nxt_ie_utt"]

            self.assertEqual(list(range(gt_ie_pos, gt_ir_pos)), pair_obj['ie_utt_pos'])
            self.assertEqual(list(range(gt_ir_pos, gt_nxt_pos)), pair_obj['ir_utt_pos'])
            self.assertEqual(list(range(gt_nxt_pos, (ir_pos + [ir_pos[-1] + 2])[ir_pos.index(gt_ir_pos) + 1])),
                             pair_obj['nxt_ie_utt_pos'])
            print(f"{conv_id}-{pair_obj['ie_utt_pos'][0]}")
            print(f"\t {ie_utt}")
            print(f"\t {ir_utt}")
            print(f"\t {nxt_ie}")

        self.assertEqual(nbr_pairs, 6)

    def test_dummy_ieir_to_df(self):
        self.dummy_interview.load_interview_data()
        df, df_ids = self.dummy_interview.filter_ieirpairs_to_df()
        print(df)
        print(df_ids)

    def test_dummy_ieir_to_csv(self):
        """
            test if dummy dataset of one interview is correctly used to generate the (ie1, ir, ie2) triples
                ie ids should be [4, 6, 8, 10, 11, 13, 15, 16]   # index 18 removed because of short utterance
        :return:
        """
        self.dummy_interview.load_interview_data()
        df_ids = self.dummy_interview.ieir_ids_to_csv(path="paraphrase/utility/fixtures/data/dummy_interview_ieir_ids.csv")
        self.assertListEqual(df_ids[self.dummy_interview.IE_IDS].tolist(), [[4], [6], [8], [10, 11], [13], [15, 16]])

    def test_dummy_id_csv_to_df(self):
        self.dummy_interview.load_interview_data()
        ieir_ids_path = "paraphrase/utility/fixtures/data/dummy_interview_ieir_ids.csv"
        df, _ = self.dummy_interview.filter_ieirpairs_to_df()
        self.dummy_interview.ieir_ids_to_csv(path=ieir_ids_path)

        constructed_df = self.dummy_interview.ieir_from_id_csv(ieir_ids_path, "fixtures/data/dummy_interview_ieir_ids.csv")
        self.assertTrue(df.equals(constructed_df))

    def test_get_utt(self):
        interview_obj = {'id': 'NPR-4', 'url': 'https://www.npr.org/templates/story/story.php?storyId=16778438', 'title': 'Washington, D.C. Facing HIV/AIDS Epidemic', 'summary': "A new study says one in 50 people in the nation's capital have AIDS, and blacks comprise more than 80 percent of new cases in the city. Farai Chideya talks to Dr. Shannon Hader, who directs Washington, D.C.'s HIV/AIDS Administration.", 'utt': ["This is NEWS & NOTES. I'm Farai Chideya.", "In the nation's capital, a killer is on the loose. It's been operating in America for decades now. We're talking about AIDS. Tomorrow is World AIDS Day. Today, we'll discuss staggering new information on how prevalent AIDS is in Washington D.C., particularly among African-Americans. Overall, the rate of AIDS cases in Washington D.C. is about 10 times higher than in the United States. Dr. Shannon Hader is the director of the D.C. HIV/AIDS Administration. Welcome.", 'Thank you.', "So, these are really some chocking numbers. Sixty percent of the city's residents are African-American, but 81 percent of new HIV cases in the city are among African-Americans. How many people are we really talking about?", "Well, you know, we have about 12,500 people in the district right now living with HIV and AIDS, but about 80 percent of those are mainly African-American communities. So, we're talking a high number of people, not just a hundred or two hundred, but thousands.", 'What about the trend lines? Are you seeing these number of new infections increase?', "Well, you know, certainly over the United States, the trend over the last decade has been increasing racial disparities and the HIV epidemic with more African-Americans affected. Here in the district, we have really good data for the last 2001 through 2006, and what we see is that we're not gaining much ground at this point in terms of reducing infections, although we seem to be holding a little bit even. And - but I think particularly among the women, the rates among women have been increasing over the last five or six years.", 'What percentage of women in the D.C. area are African-American who were infected?', 'Mm-hmm. Among all the women that we know are infected with HIV in the district, about 90 percent of them are African-American.', 'With these numbers, with the racial disparities, what is being done? What are the approaches that you and other government, public health officials, nonprofits are taking to really start addressing this?', "Well, I think what we're doing and what we need to continue to do is an attack on all fronts. First step is, information is power. These data, these hard facts give us a good picture for everyone at the individual level, at the community level, at the government level, at the policy level, to really wake up if they haven't and see the nature of the epidemic we're dealing with. Second, it's about services and it's about taking action, both to protect yourself and protect others. We are ramping what was already sort of a groundbreaking HIV policy in the district, which is this know your status, HIV tests should be just the same as knowing about your other routine health indicators.", "So, our goal is, by 2009, when you go to an emergency department, they should routinely offer you an HIV test. When you show up at your primary care doctor's office, you should get, just like you get the rest of the tests for your annual physical - you get your BMI for obesity, you get a blood pressure for hypertension, you get your blood sugar for diabetes - you should be getting your HIV status as well, without having to sort of beg for it or ask specifically. This has to be part and parcel about how we all approach our general health going forward.", "There have been celebrity campaigns that say things like, it's good know, know your status, et cetera, et cetera, et cetera, but people are afraid. All of us have fears and some people may not want to know. What's the sense that you get of that?", "Well, I think that that issue of stigma, fear, and silence is huge. And absolutely, that impacts people searching their test results, but it also impacts people taking preventive measures and taking care of measures to keep their health strong. I've been incredibly motivated by Mayor Fenty's leadership in saying, I'm making HIV/AIDS our number one health priority here in the district. And, in large part, a lot of that has to do with saying, come on, let's come together, let's break the stigma, break the fear, break the silence.", "Who's really responsible for this - responsible may be the wrong word, but, I mean, Washington D.C. is a very interesting case of the overlap of the federal government and the local government. So, what responsibilities does it seem as if each has in dealing with this issue?", "Well, you know what, we're all responsible and we have to use all the resources that are out there, whether they're district or federal, to get to the next level of our HIV response. Certainly, one of the specific relationship issues that's come out in D.C. has been this issue of Congress limiting our ability to spend our own district tax money on our own district programs and specifically, I'm talking about needle exchange programs.", "Certainly, Congresswoman Eleanor Holmes Norton has been working as well as Mayor Fenty has been working to convince Congress to remove that restriction on our funds, and I'm confident that that's going to happen this year. So, that's something that's specific to the district that other jurisdictions don't have to deal with.", 'How much of needle exchange programs become more popular? They were extremely controversial when they were first proposed and first implemented.', 'Mm-hmm.', 'Is this now a fairly accepted form of a public health intervention?', "Well, I think when it comes to comprehensive substance abuse, HIV prevention, we want a full toolkit available. Needle exchange is just one element in that full toolkit, and a lot of the wraparound services - including having on-demand treatment access for drug cessation, including having medical care available, including mental health services available, including having prevention information going out, those are all part of the toolkit. So, we don't want just one tool of the toolkit or just another tool in the toolkit, we want the whole thing at our disposal to really have a comprehensive program.", 'Well, Dr. Hader. Thanks for the information.', "Well, thank you for helping share that information. I think this really important and I hope a lot of your audience doesn't just listen, but takes the topic home, starts breaking that silence and stigma, and have some dinner-table conversations.", "Well, thanks again. Dr. Shannon Hader, she's the director D.C. HIV/AIDS Administration."], 'speaker': ['FARAI CHIDEYA, host', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host', 'Dr. SHANNON HADER (Director, D.C. HIV/AIDS Administration)', 'FARAI CHIDEYA, host']}
        list_ids = '[1]'
        utt = MediaSumProcessor.get_utt(interview_obj, list_ids)
        self.assertFalse("\n" in utt)
        print(utt)

    def test_load_split_json(self):
        split = self.dummy_interview.load_split_data("fixtures/output/dummy_split.json")
        self.assertIn("train", split.keys())
        self.assertIn("val", split.keys())
        self.assertIn("test", split.keys())

    " --------- DUMMY SAMPLING TESTING ------------- "
    def test_traindevtest_split(self):
        to_split = [f"NPR-{i}" for i in range(20)]
        seed = 13497754789
        train_ids, dev_ids, test_ids = self.interview._train_dev_test_listsplit(to_split, seed)
        self.assertEqual(14, len(train_ids))
        self.assertEqual(3, len(dev_ids))
        self.assertEqual(3, len(test_ids))

        # make sure the same list is generated from the same SEED
        train_ids_2, _, _ = self.interview._train_dev_test_listsplit(to_split, seed)
        self.assertEqual(train_ids, train_ids_2)

    def test_dummy_genpairs(self):
        """
            test: train/dev/test split of the triples of the 1 interview in the respective csvs
        :return:
        """
        self.dummy_interview.filter_and_shuffle_ghpairs_to_tsv(ghID_path="paraphrase/utility/fixtures/output/dummy_train_ieir_ids.csv")

    def test_dummy_iter_sample_pairs(self):

        for q_id, ie_utt, ir_utt, summary in self.dummy_interview\
                .iter_sample_pairs(2, "fixtures/output/dummy_train_ieir_ids.csv"):
            print(q_id, ie_utt, ir_utt, summary)

    def test_dummy_iter_for_annotation(self):

        for q_id, ie_utt, ir_utt, summary in self.dummy_interview. \
                iter_for_annotation("fixtures/output/dummy_train_ieir_ids.csv"):
            print(q_id, ie_utt, ir_utt, summary)


    " ------------ MediaSum TESTING ----------------- "

    def test_iter_for_annotation(self):
        i = 0

        for q_id, ie_utt, ir_utt, summary in self.interview. \
                iter_for_annotation("fixtures/output/_train_ieir_ids.csv"):
            print(q_id, ie_utt, ir_utt, summary)
            i += 1
            if i > 20:
                break

    def test_load_complete_json(self):
        self.interview.load_interview_data()
        self.assertListEqual(list(self.interview.interviews.columns),
                             ['id', 'program', 'date', 'url', 'title', 'summary',
                              'utt', 'speaker'])
        self.assertEqual(self.interview.interviews.shape[0],
                         49420 + 414176)  # see: https://aclanthology.org/2021.naacl-main.474.pdf



    def test_print_convo(self):
        self.interview.print_convo(json.loads(self.convo_2p))

    def test_filtered2person(self):
        self.interview.filter_2_person()  # 23,714 2-person interviews in NPR + ? in CNN
        # DOES NOT PASS THIS TEST:
        # self.assertGreater(self.interview.interviews.shape[0], 23714)
        # 14038 2-person NPR vs. 14547 2-person NPR with non-unqiue --> either/guest host names non-unique or INTERVIEW not included
        print(f"found {self.interview.interviews.shape[0]} 2-person interviews")
        self.assertEqual(len(self.interview.interviews.shape[0]), 66863)
        for index, interview in self.interview.interviews.iterrows():
            self.assertTrue(self.interview._is2p(interview))

    def test_sample_convos(self):
        """
            test the sample_convos function of the MediaSumProcessor class
        :return:
        """
        self.interview.filter_2_person()
        sample_interviews = self.interview.sample_convos(10, min_len=30)
        self.assertEqual(10, sample_interviews.shape[0])
        for _, interview in sample_interviews.iterrows():
            self.interview.print_convo(interview)




    def test_ieir_pairs(self):

        # get iterator over interviews with 2 persons
        self.interview.load_interview_data()
        self.interview.filter_2_person()
        row_iterator = self.interview.interviews.iterrows()

        nbr_interviews = 0
        prev_id = -1
        i = 0

        # get iterator that we want to test
        pair_gen = self.interview._iter_filtered_instances()
        npr = 0
        npr_triples = 0
        cnn = 0
        cnn_triples = 0

        for pair_obj in pair_gen:  # iterate over all the ieir pairs

            conv_id = pair_obj["conv_id"]
            ie_utt = pair_obj["ie_utt"]
            ir_utt = pair_obj["ir_utt"]
            ie_rply_utt = pair_obj["nxt_ie_utt"]

            if i < 100:
                print(conv_id)
                print(f"\t {ie_utt}")
                print(f"\t {ir_utt}")
                print(f"\t {ie_rply_utt}")
                i += 1

            if conv_id == "NPR-31761":
                print("Tada")

            if prev_id == -1 or conv_id != prev_id:

                # Make sure that the interview has exactly 2 speakers and at least 6 turns
                found_2p_interview = False
                guest_found = False
                while not found_2p_interview:
                    shifts = 0
                    _, cur_interview = next(row_iterator)
                    speaker_list = self.interview._make_names_unique(cur_interview["speaker"])
                    if len(set(speaker_list)) > 2:
                        continue
                    first_shift = 0
                    for k in range(1, len(speaker_list)):
                        # [Host, Guest, H, G, H, G, H, G] --> at least 7 shifts to get one triple
                        #      [Guest, H, G, H, G, H, G] --> remove welcome
                        #               [ G, H, G, H, G] --> remove non-reacted reply
                        #               [ G, H, G, H] --> remove second to last guest (likely goodbye)
                        #               [ G, H, G]
                        if shifts < 7 and speaker_list[k] != speaker_list[k - 1]:
                            shifts += 1
                        if shifts == 3:
                            first_shift = k
                        elif shifts >= 7:
                            break
                    if len(set(speaker_list)) == 2 and shifts >= 7:
                        if shifts == 7 and len(cur_interview["utt"][first_shift].split(" ")) <= 2:
                            continue
                        found_2p_interview = True

                nbr_interviews += 1
                speaker_list = self.interview._make_names_unique(cur_interview["speaker"])
                prev_id = cur_interview["id"]
                if "NPR" in prev_id:
                    npr += 1
                else:
                    cnn += 1

            if "NPR" in cur_interview["id"]:
                npr_triples += 1
            else:
                cnn_triples += 1

            self.assertIsInstance(conv_id, str)
            self.assertEqual(cur_interview["id"], conv_id)
            # utterances are in the utterance list
            self.assertIn(ie_utt[0], cur_interview["utt"])
            self.assertIn(ir_utt[0], cur_interview["utt"])
            # the speaker of the interviewee utterance is unequal to the first utterance
            self.assertNotEqual(speaker_list[pair_obj["ie_utt_pos"][0]], speaker_list[0])
            # the speaker of the interviewer utterance is the same as the first speaker in the interview
            self.assertEqual(speaker_list[pair_obj["ir_utt_pos"][0]], speaker_list[0])

        print(f"{nbr_interviews} interviews were found with 2-person setup and INTERVIEWEE-HOST pairs")
        print(f"    \t out of which {cnn} cnn and {npr} npr interviews")
        print(f"    \t out of which {cnn_triples} cnn triples and {npr_triples} npr triples interviews")

    def test_complete2p_to_csv(self):
        # GENERATE THE DATA
        self.interview.filter_2_person()
        self.interview.ieir_ids_to_csv(path="paraphrase/utility/fixtures/data/_all_interview_ieir_ids.csv")

        npr = 0
        npr_triples = 0
        cnn = 0
        cnn_triples = 0
        prev_id = 0
        nbr_interviews = 0
        for pair_obj in self.interview._iter_filtered_instances():
            conv_id = pair_obj["conv_id"]
            if "NPR" in conv_id:
                npr_triples += 1
            else:
                cnn_triples += 1
            if conv_id != prev_id:
                if "NPR" in conv_id:
                    npr += 1
                else:
                    cnn += 1
                nbr_interviews += 1
            prev_id = conv_id

        print(f"{nbr_interviews} interviews were found with 2-person setup and INTERVIEWEE-HOST pairs")
        print(f"    \t out of which {cnn} cnn and {npr} npr interviews")
        print(f"    \t out of which {cnn_triples} cnn triples and {npr_triples} npr triples interviews")
        print(f"    \t {self.interview.filtered_ie} pairs were removed due to short IE utterances")
        print(f"    \t and {self.interview.filtered_ir} pairs were removed due to short IR utterances")
        print(f"    \t and {self.interview.filtered_address} interviews were removed due to too few addresses")





    def test_genpairs_complete2p_split(self):
        seed = 13497754789
        self.interview.filter_and_shuffle_ghpairs_to_tsv(ghID_path="paraphrase/utility/fixtures/output/_train_ieir_ids.csv", seed=seed)

    def test_get_interview_from_id(self):
        npr7 = "CNN-28984"
        qid = "NPR-733"  # "CNN-13148"
        interview = self.interview.get_interview_from_id(qid)
        print(interview["speaker"])
        print(interview["url"])
        print(interview["date"])
        print(interview["summary"])
        # npr31163 = "NPR-31163"
        # interview = self.interview.get_interview_from_id(npr31163)
        # print(interview["url"])
        # npr7 = "CNN-69241"
        # interview = self.interview.get_interview_from_id(npr7)
        # print(interview["speaker"])
        # print(interview["title"])
        # self.assertEqual(npr7, interview["id"])
        # fail_id = "FAIL-ID"
        # self.assertRaises(ValueError, self.interview.get_interview_from_id, fail_id)
        # npr7 = "NPR-1462"
        # interview = self.interview.get_interview_from_id(npr7)
        # print(interview["url"])
        # print(interview["title"])

    def test_get_pair(self):
        pair = self.interview.get_qdict_from_qid('CNN-67148-13')
        # TODO: test if this is extracting the right strings
        print(pair)

    def test_iter_sample_pairs(self):
        interview = MediaSumProcessor()

        for q_id, ie_utt, ir_utt, summary in interview.iter_sample_pairs(10, "fixtures/data/_train_ieir_pairs.csv"):
            print(q_id, ie_utt, ir_utt, summary)

    def test_find_qid(self):
        interview = MediaSumProcessor()
        for q_id, ie_utt, ir_utt, summary in interview.iter_all_qs(IDS_CSV):  # if this doesn't return all try: ieir_from_csv
            if any(utt in ie_utt
                   for utt in ["there are no constitutional protections for you.",
                               "have indicated that they have been getting cooperation from the people involved",
                               "major artists like Joe Buttons and stuff",
                               "he was very pleasant"]):
                print(q_id, ie_utt, ir_utt, summary)
            elif any(utt in ir_utt
                     for utt in ["Are you ready to see that go up without any strings attached",
                                 "people on the ground don"]):
                print(q_id, ie_utt, ir_utt, summary)


