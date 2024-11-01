"""
    This file contains the prompt templates and few-shot examples

    Called with prompt_template.format((sentence1=q_item["guest utterance"],
                                                  sentence2=q_item["host utterance"],
                                                  options_="OPTIONS:\n-Yes\n-No",
                                                  summary=q_item["summary"],
                                                  date=q_item["date"],
                                                  host_name=q_item["host name"],
                                                  guest_name=q_item["guest name"])
"""

FS_PROMPT_TEMPLATES = [
    # 1. taken from annotator training/instructions
    #   - change: "In the reply, the host is paraphrasing something specific the guest says." to
    #       "Does the host paraphrase or repeat something specific the guest says?"
    {
        "prefix": "A Paraphrase is a rewording or repetition of content in the guest's statement. "
                  "It rephrases what the guest said.",
        "item": "Given an interview on {date} with the summary: {summary}\n"
                "Guest and Host say the following:\n"
                "Guest ({guest_name}): {guest_utterance}\n"
                "Host ({host_name}): {host_utterance}",
        "prompt": "In the reply, does the host paraphrase something specific the guest says?",
        "reply": "Explanation: {explanation}\n"
                 "Verbatim Quote Guest: {quote_guest}\n"
                 "Verbatim Quote Host: {quote_host}\n"
                 "Classification: {classification}",
        "quote_string": "\"{quote}\"",
        "start_response": "Explanation: Let's think step by step.\n"
    },
]

FEW_SHOT_EXAMPLES = [  # taken from annotator instructions
    # 1. Introductory definition example: Paraphrase
    {
        "date": "-",  # CNN-357405-7
        "summary": "Fresh Prince Star Alfonso Ribeiro Sues Over Dance Moves; "
                   "Rapper 2 Milly Alleges His Dance Moves were Copied.",
        "guest name": "TERRENCE FERGUSON, RAPPER",
        "host name": "QUEST",
        "guest utterance": "I guess it was season 5 when they premiered it in the game. A bunch of DMs, a bunch of "
                           "Twitter requests, e-mails, everything was like, you, your game is in the dance, you need to sue, "
                           "\"Fortnite\" stole it. Even like big artists, major artists like Joe Buttons and stuff, "
                           "they have their own like show, daily struggle, they say, you, you must sue \"Fortnite\", "
                           "and I'm like, \"Fortnite\", what is that? I don't even know what it is --",
        "host utterance": "So you weren't even familiar?",
        "Explanation": "Let's think step by step.\n"
                       "Terrence Ferguson says at the end of his turn that he didn't know Fortnite.\n"
                       "Quest, the host of the interview, repeats that the guest doesn't know Fortnite.\n"
                       "So they both say that the guest didn't know Fortnite. "
                       "Therefore, the answer is yes, the host is paraphrasing the guest.",
        "Classification": "Yes.",
        "Quote Guest": ["I'm like, \"Fortnite\", what is that?  I don't even know what it is"],
        "Quote Host": ["you weren't even familiar?"]
    },
    # 2. Comprehension Check example: Paraphrase
    {
        "date": "2013-10-1",  # CNN-215723-3
        "summary": "Interview With Idaho Congressman Raul Labrador",
        "guest name": "REP. RAUL LABRADOR (R), IDAHO",
        "host name": "BLITZER",
        "guest utterance": "That's what we have been asking the president. "
                           "We would like the senators to actually come and negotiate with us. "
                           "So I think that would be a terrific idea.",
        "host utterance": "You say you want to negotiate, but what about the debt ceiling? "
                           "Are you ready to see that go up without any strings attached, as the president demands?",
        "Explanation": "Let's think step by step.\n"
                       "Republican Raul Labrador says that \"we\" want to negotiate with the senators. By that he probably means the Republicans.\n"
                       "Blitzer, the host of the interview, says that \"you\" want to negotiate. Blitzer probably means the Republicans as well.\n"
                       "So both of them are saying that the Republicans want to negotiate. "
                       "Therefore, the answer is yes, host is paraphrasing the guest.",
        "Classification": "Yes.",
        "Quote Guest": ["We would like the senators to actually come and negotiate with us."],
        "Quote Host": ["you want to negotiate"]
    },
    # 3. Additional Meaning example: No Paraphrase
    {
        "date": "2015-12-15",  # CNN-271388-12
        "summary": "Interview with Kentucky Senator Rand Paul",
        "guest name": "SEN. RAND PAUL (R-KY), PRESIDENTIAL CANDIDATE",
        "host name": "TAPPER",
        "guest utterance": "If you're not in our country, there are no constitutional protections for you.",
        "host utterance": "So, you don't have a problem with Facebook giving the government access to the private "
                          "accounts of people applying to enter the U.S.?",
        "Explanation": "Let's think step by step.\n"
                       "Rand Paul says that there are no constitutional protections for people outside \"our\" country. "
                       "Since Rand Paul was a Kentucky Senator in 2015, he is referring to the United States.\n"
                       "The host of the interview, Tapper, asks if Paul has no problem with Facebook "
                       "giving the U.S. government access to accounts of people who apply to enter the United States.\n "
                       "While Tapper and Paul both talk about people outside the U.S., "
                       "Paul talks about constitutional protections in the U.S. and "
                       "Tapper infers Paul's opinion on a company giving the government access to private accounts of their users.\n"
                       "Tapper, the host, adds a conclusion to what the guest said (\"so you don't have a problem with ...\") "
                       "without rewording or repeating the content of the guest's utterance. "
                       "Therefore, the answer is no, the host does not reword or repeat the guest.",
        "Classification": "No.",
        "Quote Guest": None,
        "Quote Host": None
    },
    # 4. several highlights example: Paraphrase
    {
        "date": "2005-7-20",  # CNN-96249-7
        "summary": "CIA Operative Talks About Life After Cover Blown",
        "guest name": "MASSIMO CALABRESI, \"TIME\" MAGAZINE",
        "host name": "PHILLIPS",
        "guest utterance": "She was very pleasant. Talked about family life. "
                           "They chatted about errands they need to run and things like that.",
        "host utterance": "Well, she talked a lot about her family and her kids. "
                          "And you get a personal sense for how they're living day by day.",
        "Explanation": "Let's think step by step.\n"
                       "Massimo Calabresi talks about a conversation with a person he refers to as \"she\". "
                       "How \"she\" seemed and what the conversation was about: Family life and errands.\n"
                       "Phillips, the interview host, also talks about that same conversation with \"she\". "
                       "That \"she\" talked about family and daily life.\n "
                       "Therefore, the answer is yes, the host is paraphrasing the guest.",
        "Classification": "Yes.",
        "Quote Guest": ["She", "Talked about family life.", "errands they need to run and things like that."],
        "Quote Host": ["she talked", "about her family and her kids.", "how they're living day by day."]
    },
    # 5. Attention Check example: No Paraphrase
    {
        "date": "2018-05-29",  # NPR-44857-6
        "summary": "Two weeks after the wave of protests and deadly clashes at the Israeli border, "
                   "many Gazans are wounded and feeling like the demonstrations didn't bring any tangible benefits.",
        "guest name": "DANIEL ESTRIN, BYLINE",
        "host name": "STEVE INSKEEP, HOST",
        "guest utterance": "And so that's the main question I've been asking people here, is, was the price worth it?",
        "host utterance": "You're telling me people on the ground don't see it that way.",
        "Explanation": "Let's think step by step.\n"
                       "Daniel Estrin says that he asked people if the price was worth it.\n "
                       "Steve Inskeep, the host of the interview, asks if people \"don't see it that way,\" "
                       "asking what people responded to Estrin's question to people.\n"
                       "However, the host does not paraphrase the question itself. "
                       "Therefore, the answer is no, the host does not reword or repeat the guest.",
        "Classification": "No.",
        "Quote Guest": None,
        "Quote Host": None
    },
    # 6. Isolation/Semantic Relatedness example: No paraphrase
    {
        "date": "2005-7-28",  # 'CNN-96467-5'
        "summary": "Pregnant Philadelphia Woman Still Missing",
        "guest name": "TONY HANSON, KYW NEWSRADIO",
        "host name": "PHILLIPS",
        "guest utterance": "Police have indicated that they have been getting cooperation from the people involved, "
                           "of course, they are looking at all of her personal relationships "
                           "to see if there were any problems there.",
        "host utterance": "I know you've talked to various members of her family. What did they tell you?",
        "Explanation": "Let's think step by step.\n"
                       "KYW Newsradio's Tony Hanson that the police are investigating \"her\" personal relationships.\n"
                       "Phillips says KYW Newsradio's Hanson has spoken to various members of \"her\" family.\n"
                       "Hanson, the guest, talks about the police conducting interviews, while, the host, Phillips,"
                       " talks about the reporter Hanson conducting interviews. "
                       "Therefore, the answer is no, "
                       "the host is not repeating the actual content of the guest's statement. "
                       "The host rather continues on with a related topic.",
        "Classification": "No.",
        "Quote Guest": None,
        "Quote Host": None
    },
    # 7. Context Example: Paraphrase
    {
        "date": "2005-5-26",
        "summary": "Lionel Tate is back in jail for allegedly holding up a pizza deliveryman. The 18-year-old "
                   "Florida youth was the youngest person to be sent to prison for life in U.S. history. At age 12, "
                   "he was accused of murdering his 6-year-old neighbor and friend Tiffany Eunick when he claimed "
                   "he was demonstrating wrestling moves. Ed Gordon talks with Sgt. DeLacy Davis from New Jersey, "
                   "a mentor and one of several supporters that put together a \"re-entry into society plan\" "
                   "for Tate.",
        "guest name": "DE LACY DAVIS",
        "host name": "ED GORDON, host",
        "guest utterance": "I think that, God willing, and then certainly if we were given another shot at this apple, "
                           "I think the entire group would be amenable to shipping him here to me, "
                           "which is what we felt would be a better environment to give him a new start. "
                           "People, places and things needed to be changed, and consistently changed, "
                           "and the plan adjusted based upon how he was faring.",
        "host utterance": "So that would be, actually, coming to New Jersey and being under the auspices, frankly, "
                          "of De Lacy Davis.",
        "Explanation": "Let's think step by step.\n"
                       "The guest, De Lacy Davis, talks about a plan to give Lionel Tate a fresh start "
                       "by sending him to De Lacy Davis (\"shipping him here to me\").  From the summary, "
                       "we know that De Lacy Davis is probably in New Jersey.\n"
                       "Ed Gordon, the interview host, clarifies this and says the plan is for Tate "
                       "to come to New Jersey and  be mentored by De Lacy Davis.\n"
                       "Both are talking about Tate being sent to New Jersey, to De Lacy Davis. "
                       "Therefore, the answer is yes, the host paraphrases the guest.",
        "Classification": "Yes.",
        "Quote Guest": ["shipping him here to me"],
        "Quote Host": ["coming to New Jersey and being under the auspices", "of De Lacy Davis."],
    },
    # 8. Comprehension Check: Paraphrase
    {
        "date": " 2000-8-3",
        "summary": "Kissinger: Bush is Fully Qualified for Foreign Policy Decisions",
        "guest name": "HENRY KISSINGER, FMR. SECRETARY OF STATE",
        "host name": "NATALIE ALLEN, CNN ANCHOR",
        "guest utterance": "No, I haven't talked to him but I've talked to his secretary and "
                           "we've passed messages back and forth between the family and me. "
                           "And they tell me he's improved a lot. And I'm to see him tomorrow morning.",
        "host utterance": "I'm sure that will cheer him up, to have a visit from you, and the doctors did say, "
                          "just a short while ago, he's expected make a full recovery.",
        "Explanation": "Let's think step by step.\n"
                       "Former Secretary of State Henry Kissinger talks about someone (\"him\") who is sick and possibly hospitalized. "
                       "Kissinger says that he will see \"him\" tomorrow morning.\n"
                       "Natalie Allen, the CNN anchor, says it will cheer \"him\" up when Kissinger visits.\n"
                       "Both the interview guest Kissinger and the interview host Allen mention that Kissinger will visit \"him\". "
                       "Therefore, the answer is yes, the host is paraphrasing the guest.",
        "Classification": "Yes.",
        "Quote Guest": ["I'm to see him."],
        "Quote Host": ["him", "have a visit from you"],
    }
]


def build_few_shot_prompt(template, examples, new_item):
    """
        template is of the form
            {
                "prefix": None if no prefix, or a string that is preappended
                "item": FORMAT FOR A SINGLE ITEM
                "prompt": FORMAT FOR THE PROMPT
                "reply": FORMAT FOR THE REPLY FOR ONE ITEM
                "quote_string": FORMAT FOR THE QUOTE STRING
            }
    :param template:
    :param examples:
    :param new_item:
    :return:
    """
    prompt = ""
    if template["prefix"] is not None:
        prompt += template["prefix"] + "\n\n"
    for example in examples:
        prompt += template["item"].format(date=example["date"],
                                          summary=example["summary"],
                                          guest_name=example["guest name"],
                                          host_name=example["host name"],
                                          guest_utterance=example["guest utterance"],
                                          host_utterance=example["host utterance"])
        prompt += "\n"
        prompt += template["prompt"] + "\n\n"
        if example["Quote Guest"] is not None:
            guest_quote_string = " ".join(
                template["quote_string"].format(quote=quote) for quote in example["Quote Guest"])
        else:
            guest_quote_string = "None."
        if example["Quote Host"] is not None:
            host_quote_string = " ".join(
                template["quote_string"].format(quote=quote) for quote in example["Quote Host"])
        else:
            host_quote_string = "None."
        prompt += template["reply"].format(explanation=example["Explanation"],
                                           classification=example["Classification"],
                                           quote_guest=guest_quote_string,
                                           quote_host=host_quote_string)
        prompt += "\n\n\n"
    prompt += template["item"].format(date=new_item["date"],
                                      summary=new_item["summary"],
                                      guest_name=new_item["guest name"],
                                      host_name=new_item["host name"],
                                      guest_utterance=new_item["guest utterance"],
                                      host_utterance=new_item["host utterance"])
    prompt += "\n"
    prompt += template["prompt"] + "\n\n"
    prompt += template["start_response"]

    return prompt


class FewShotPattern:  # copied from https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py
    """Patterns for few-shot tasks.

      The few-shot input are composed by a few examplers followed by final_suffix:
      {exampler no. 1} + {exampler no. 2} + {exampler no. 3}... + {final_suffix}

      Each exampler has the following format:
      {inputs_prefix} + {inputs} + {x_y_delimiter} + {targets_prefix} + {targets} +
      {example_separator}
    """

    def __init__(self, inputs: str, targets: str, inputs_prefix: str = "", targets_prefix: str = "",
                 x_y_delimiter: str = "\n\n", example_separator: str = "\n\n\n", final_suffix: str = "",
                 input_pattern: str = "{{inputs}}{final_suffix}", in_template_mix: bool = True):
        self.inputs = inputs
        self.targets = targets
        self.inputs_prefix = inputs_prefix
        self.targets_prefix = targets_prefix
        self.x_y_delimiter = x_y_delimiter
        self.example_separator = example_separator
        self.final_suffix = final_suffix
        self.input_pattern = input_pattern
        self.in_template_mix = in_template_mix

    @property
    def few_shot_kwargs(self):
        return dict(
            inputs_prefix=self.inputs_prefix,
            targets_prefix=self.targets_prefix,
            x_y_delimiter=self.x_y_delimiter,
            example_separator=self.example_separator,
            final_suffix=self.final_suffix,
            input_pattern=self.input_pattern)

    @property
    def combined_inputs(self):
        return self.inputs_prefix + self.inputs + self.x_y_delimiter

    @property
    def combined_targets(self):
        return self.targets_prefix + self.targets + self.example_separator

    @property
    def combined_inputs_w_target_prefix(self):
        return self.inputs_prefix + self.inputs + self.x_y_delimiter + (
            self.targets_prefix)

    @property
    def combined_targets_wo_target_prefix(self):
        return self.targets + self.example_separator
