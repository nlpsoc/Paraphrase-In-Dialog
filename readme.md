# What's Mine becomes Yours: Detecting Context-Dependent Paraphrases in News Interview Dialogs

Great that you are here. This is the repository for our EMNLP 2024 main conference paper "What's Mine becomes Yours: Detecting Context-Dependent Paraphrases in News Interview Dialogs", see the preprint [on arxiv](https://arxiv.org/abs/2404.06670). You probably are either interested in the **annotated data** (https://huggingface.co/datasets/AnnaWegmann/Paraphrases-in-Interviews) or using a **computational model to predict paraphrases in dialog** (see https://huggingface.co/AnnaWegmann/Highlight-Paraphrases-in-Dialog-ALL for the ALL model and https://huggingface.co/AnnaWegmann/Highlight-Paraphrases-in-Dialog for the AGGREGATED model).


## Annotation Data

You can find the annotation data among the huggingface datasets (https://huggingface.co/datasets/AnnaWegmann/Paraphrases-in-Interviews). We share it with a research-only license.

Data is tab separated and looks like

```
QID	Annotator	Session	Is Paraphrase	Guest Tokens	Guest Highlights	Host Tokens	Host Highlights
CNN-177596-7	PROLIFIC_1	R_2PoEZfAptrkFdsx	0	This is not good.	[0, 0, 0, 0]	This is what you don't want happening with your menorah, folks.	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
CNN-177596-7	PROLIFIC_2	R_3HCjJuW3mB9PQpL	1	This is not good.	[1, 1, 1, 1]	This is what you don't want happening with your menorah, folks.	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
```

You might want to get the votes for one question id: 

```python
from datasets import load_dataset
import pandas as pd
import ast

# Load the dataset
dataset = load_dataset("AnnaWegmann/Paraphrases-in-Interviews")

# Convert the dataset to a pandas DataFrame
split_names = list(dataset.keys())
dataframes = [dataset[split].to_pandas() for split in split_names]
df = pd.concat(dataframes, ignore_index=True)  # if you just need one split: dataset['train'].to_pandas()

# Specify the QID to process (e.g., 'CNN-177596-7'), it has to be in the correct split
qid = 'CNN-80522-7'

# Filter the DataFrame for the specified QID
group = df[df['QID'] == qid]

# Compute total votes and paraphrase votes
total_votes = len(group)
paraphrase_votes = group['Is Paraphrase'].astype(int).sum()

# Process Guest Highlights
guest_highlights_list = group['Guest Highlights'].apply(ast.literal_eval).tolist()
guest_highlights_sums = [sum(x) for x in zip(*guest_highlights_list)]

# Process Host Highlights
host_highlights_list = group['Host Highlights'].apply(ast.literal_eval).tolist()
host_highlights_sums = [sum(x) for x in zip(*host_highlights_list)]

# Output the results
print(f"QID: {qid}")
print(f"Total Votes: {total_votes}")
print(f"Paraphrase Votes: {paraphrase_votes}\n")

print("Guest Tokens:")
print(group['Guest Tokens'].iloc[0])
print("Guest Highlights Counts:")
print(guest_highlights_sums, "\n")

print("Host Tokens:")
print(group['Host Tokens'].iloc[0])
print("Host Highlights Counts:")
print(host_highlights_sums)
```

should return

```
QID: CNN-80522-7
Total Votes: 3
Paraphrase Votes: 3

Guest Tokens:
It's a positive sign, I think. I was encouraged to see that. And people always prefer, of course, to see the pope as the principal celebrant of the mass. So that's good. That'll be tonight. And it will be his 26th mass and it will be the 40th or, rather, the 30th time that this is offered in round the world transmission. And it will be my 20th time in doing it as a television commentator from Rome so.
Guest Highlights Counts:
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0] 

Host Tokens:
Yes, you've been doing this for a while now.
Host Highlights Counts:
[0, 3, 3, 3, 3, 3, 3, 3, 3]
```


## Modeling

### Token Classifier

Use our trained token classifier on the huggingface hub. 

**ALL model**: Trained on all annotations in the training set. Has higher precision. Might be better out-of-domain from few manual tests. See: https://huggingface.co/AnnaWegmann/Highlight-Paraphrases-in-Dialog-ALL

**AGGREGATED model**: Trained on aggregated annotations of the training set. Has higher recall. See: https://huggingface.co/AnnaWegmann/Highlight-Paraphrases-in-Dialog 

These models were trained to predict the label of a word based on the first token of a word. The label is 1 if the word is part of a paraphrase and 0 if it is not. 

We give a contained example to use the model on an exchange. To use our scripts, see `token_classifier.py`.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class ParaphraseHighlighter:
    def __init__(self, model_name="AnnaWegmann/Highlight-Paraphrases-in-Dialog-ALL"):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Get the label id for 'LABEL_1'
        self.label2id = self.model.config.label2id
        self.label_id = self.label2id['LABEL_1']
    
    def highlight_paraphrase(self, text1, text2):
        # Tokenize the inputs with the tokenizer
        encoding = self.tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True)
        
        outputs = self.model(**encoding)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, num_labels)
        # Apply softmax to get probabilities, automatically places [SEP] token
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Shape: (batch_size, sequence_length, num_labels)
        
        # Convert token IDs back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])        
        # Get word IDs to map tokens to words
        word_ids = encoding.word_ids(batch_index=0)
        # Get sequence IDs to know which text the token belongs to
        sequence_ids = encoding.sequence_ids(batch_index=0)
        
        # Collect words and probabilities for each text
        words_text1 = []
        words_text2 = []
        probs_text1 = []
        probs_text2 = []
        
        previous_word_idx = None
        
        # For determining if there are high-probability words in both texts
        has_high_prob_text1 = False
        has_high_prob_text2 = False
        
        for idx, (word_idx, seq_id) in enumerate(zip(word_ids, sequence_ids)):
            if word_idx is None:
                # Skip special tokens like [CLS], [SEP], [PAD]
                continue

            if word_idx != previous_word_idx:
                # Start of a new word
                word_tokens = [tokens[idx]]

                # Get the probability for LABEL_1 for the first token of the word
                prob_LABEL_1 = probs[0][idx][self.label_id].item()

                # Collect subsequent tokens belonging to the same word
                j = idx + 1
                while j < len(word_ids) and word_ids[j] == word_idx:
                    word_tokens.append(tokens[j])
                    j += 1

                # Reconstruct the word
                word = self.tokenizer.convert_tokens_to_string(word_tokens).strip()

                # Check if probability >= 0.5 to uppercase
                if prob_LABEL_1 >= 0.5:
                    word_display = word.upper()
                    if seq_id == 0:
                        has_high_prob_text1 = True
                    elif seq_id == 1:
                        has_high_prob_text2 = True
                else:
                    word_display = word

                # Append the word and probability to the appropriate list
                if seq_id == 0:
                    words_text1.append(word_display)
                    probs_text1.append(prob_LABEL_1)
                elif seq_id == 1:
                    words_text2.append(word_display)
                    probs_text2.append(prob_LABEL_1)
                else:
                    # Should not happen
                    pass

            previous_word_idx = word_idx
        
        # Determine if there are words in both texts with prob >= 0.5
        if has_high_prob_text1 and has_high_prob_text2:
            print("is a paraphrase")
        else:
            print("is not a paraphrase")
        
        # Function to format and align words and probabilities
        def print_aligned(words, probs):
            # Determine the maximum length of words for formatting
            max_word_length = max(len(word) for word in words)
            # Create format string for alignment
            format_str = f'{{:<{max_word_length}}}'
            # Print words
            for word in words:
                print(format_str.format(word), end=' ')
            print()
            # Print probabilities aligned below words
            for prob in probs:
                prob_str = f"{prob:.2f}"
                print(format_str.format(prob_str), end=' ')
            print('\n')
        
        # Print text1's words and probabilities aligned
        print("\nSpeaker 1:")
        print_aligned(words_text1, probs_text1)
        
        # Print text2's words and probabilities aligned
        print("Speaker 2:")
        print_aligned(words_text2, probs_text2)
        
# Example usage
highlighter = ParaphraseHighlighter()
text1 = "And it will be my 20th time in doing it as a television commentator from Rome so."
text2 = "Yes, you've been doing this for a while now."
highlighter.highlight_paraphrase(text1, text2)
```

should return

```
is a paraphrase

Speaker 1:
And         it          will        be          my          20TH        TIME        IN          DOING       IT          as          a           television  commentator from        Rome        so.         
0.06        0.38        0.35        0.37        0.45        0.60        0.51        0.51        0.51        0.59        0.38        0.37        0.42        0.38        0.24        0.26        0.14        

Speaker 2:
Yes,   YOU'VE BEEN   DOING  THIS   FOR    A      WHILE  now.   
0.07   0.60   0.65   0.63   0.68   0.62   0.60   0.64   0.48   
```

### In-Context Learning

We used the following prompt on our dataset with the most success. This is very much specific to the interview setting. We have not yet tried variations for other types of dialog data.

```
A Paraphrase is a rewording or repetition of content in the guest's statement. It rephrases what the guest said.

Given an interview on - with the summary: Fresh Prince Star Alfonso Ribeiro Sues Over Dance Moves; Rapper 2 Milly Alleges His Dance Moves were Copied.
Guest and Host say the following:
Guest (TERRENCE FERGUSON, RAPPER): I guess it was season 5 when they premiered it in the game. A bunch of DMs, a bunch of Twitter requests, e-mails, everything was like, you, your game is in the dance, you need to sue, "Fortnite" stole it. Even like big artists, major artists like Joe Buttons and stuff, they have their own like show, daily struggle, they say, you, you must sue "Fortnite", and I'm like, "Fortnite", what is that? I don't even know what it is --
Host (QUEST): So you weren't even familiar?
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
Terrence Ferguson says at the end of his turn that he didn't know Fortnite.
Quest, the host of the interview, repeats that the guest doesn't know Fortnite.
So they both say that the guest didn't know Fortnite. Therefore, the answer is yes, the host is paraphrasing the guest.
Verbatim Quote Guest: "I'm like, "Fortnite", what is that?  I don't even know what it is"
Verbatim Quote Host: "you weren't even familiar?"
Classification: Yes.


Given an interview on 2013-10-1 with the summary: Interview With Idaho Congressman Raul Labrador
Guest and Host say the following:
Guest (REP. RAUL LABRADOR (R), IDAHO): That's what we have been asking the president. We would like the senators to actually come and negotiate with us. So I think that would be a terrific idea.
Host (BLITZER): You say you want to negotiate, but what about the debt ceiling? Are you ready to see that go up without any strings attached, as the president demands?
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
Republican Raul Labrador says that "we" want to negotiate with the senators. By that he probably means the Republicans.
Blitzer, the host of the interview, says that "you" want to negotiate. Blitzer probably means the Republicans as well.
So both of them are saying that the Republicans want to negotiate. Therefore, the answer is yes, host is paraphrasing the guest.
Verbatim Quote Guest: "We would like the senators to actually come and negotiate with us."
Verbatim Quote Host: "you want to negotiate"
Classification: Yes.


Given an interview on 2015-12-15 with the summary: Interview with Kentucky Senator Rand Paul
Guest and Host say the following:
Guest (SEN. RAND PAUL (R-KY), PRESIDENTIAL CANDIDATE): If you're not in our country, there are no constitutional protections for you.
Host (TAPPER): So, you don't have a problem with Facebook giving the government access to the private accounts of people applying to enter the U.S.?
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
Rand Paul says that there are no constitutional protections for people outside "our" country. Since Rand Paul was a Kentucky Senator in 2015, he is referring to the United States.
The host of the interview, Tapper, asks if Paul has no problem with Facebook giving the U.S. government access to accounts of people who apply to enter the United States.
 While Tapper and Paul both talk about people outside the U.S., Paul talks about constitutional protections in the U.S. and Tapper infers Paul's opinion on a company giving the government access to private accounts of their users.
Tapper, the host, adds a conclusion to what the guest said ("so you don't have a problem with ...") without rewording or repeating the content of the guest's utterance. Therefore, the answer is no, the host does not reword or repeat the guest.
Verbatim Quote Guest: None.
Verbatim Quote Host: None.
Classification: No.


Given an interview on 2005-7-20 with the summary: CIA Operative Talks About Life After Cover Blown
Guest and Host say the following:
Guest (MASSIMO CALABRESI, "TIME" MAGAZINE): She was very pleasant. Talked about family life. They chatted about errands they need to run and things like that.
Host (PHILLIPS): Well, she talked a lot about her family and her kids. And you get a personal sense for how they're living day by day.
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
Massimo Calabresi talks about a conversation with a person he refers to as "she". How "she" seemed and what the conversation was about: Family life and errands.
Phillips, the interview host, also talks about that same conversation with "she". That "she" talked about family and daily life.
 Therefore, the answer is yes, the host is paraphrasing the guest.
Verbatim Quote Guest: "She" "Talked about family life." "errands they need to run and things like that."
Verbatim Quote Host: "she talked" "about her family and her kids." "how they're living day by day."
Classification: Yes.


Given an interview on 2018-05-29 with the summary: Two weeks after the wave of protests and deadly clashes at the Israeli border, many Gazans are wounded and feeling like the demonstrations didn't bring any tangible benefits.
Guest and Host say the following:
Guest (DANIEL ESTRIN, BYLINE): And so that's the main question I've been asking people here, is, was the price worth it?
Host (STEVE INSKEEP, HOST): You're telling me people on the ground don't see it that way.
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
Daniel Estrin says that he asked people if the price was worth it.
 Steve Inskeep, the host of the interview, asks if people "don't see it that way," asking what people responded to Estrin's question to people.
However, the host does not paraphrase the question itself. Therefore, the answer is no, the host does not reword or repeat the guest.
Verbatim Quote Guest: None.
Verbatim Quote Host: None.
Classification: No.


Given an interview on 2005-7-28 with the summary: Pregnant Philadelphia Woman Still Missing
Guest and Host say the following:
Guest (TONY HANSON, KYW NEWSRADIO): Police have indicated that they have been getting cooperation from the people involved, of course, they are looking at all of her personal relationships to see if there were any problems there.
Host (PHILLIPS): I know you've talked to various members of her family. What did they tell you?
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
KYW Newsradio's Tony Hanson that the police are investigating "her" personal relationships.
Phillips says KYW Newsradio's Hanson has spoken to various members of "her" family.
Hanson, the guest, talks about the police conducting interviews, while, the host, Phillips, talks about the reporter Hanson conducting interviews. Therefore, the answer is no, the host is not repeating the actual content of the guest's statement. The host rather continues on with a related topic.
Verbatim Quote Guest: None.
Verbatim Quote Host: None.
Classification: No.


Given an interview on 2005-5-26 with the summary: Lionel Tate is back in jail for allegedly holding up a pizza deliveryman. The 18-year-old Florida youth was the youngest person to be sent to prison for life in U.S. history. At age 12, he was accused of murdering his 6-year-old neighbor and friend Tiffany Eunick when he claimed he was demonstrating wrestling moves. Ed Gordon talks with Sgt. DeLacy Davis from New Jersey, a mentor and one of several supporters that put together a "re-entry into society plan" for Tate.
Guest and Host say the following:
Guest (DE LACY DAVIS): I think that, God willing, and then certainly if we were given another shot at this apple, I think the entire group would be amenable to shipping him here to me, which is what we felt would be a better environment to give him a new start. People, places and things needed to be changed, and consistently changed, and the plan adjusted based upon how he was faring.
Host (ED GORDON, host): So that would be, actually, coming to New Jersey and being under the auspices, frankly, of De Lacy Davis.
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
The guest, De Lacy Davis, talks about a plan to give Lionel Tate a fresh start by sending him to De Lacy Davis ("shipping him here to me").  From the summary, we know that De Lacy Davis is probably in New Jersey.
Ed Gordon, the interview host, clarifies this and says the plan is for Tate to come to New Jersey and  be mentored by De Lacy Davis.
Both are talking about Tate being sent to New Jersey, to De Lacy Davis. Therefore, the answer is yes, the host paraphrases the guest.
Verbatim Quote Guest: "shipping him here to me"
Verbatim Quote Host: "coming to New Jersey and being under the auspices" "of De Lacy Davis."
Classification: Yes.


Given an interview on  2000-8-3 with the summary: Kissinger: Bush is Fully Qualified for Foreign Policy Decisions
Guest and Host say the following:
Guest (HENRY KISSINGER, FMR. SECRETARY OF STATE): No, I haven't talked to him but I've talked to his secretary and we've passed messages back and forth between the family and me. And they tell me he's improved a lot. And I'm to see him tomorrow morning.
Host (NATALIE ALLEN, CNN ANCHOR): I'm sure that will cheer him up, to have a visit from you, and the doctors did say, just a short while ago, he's expected make a full recovery.
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
Former Secretary of State Henry Kissinger talks about someone ("him") who is sick and possibly hospitalized. Kissinger says that he will see "him" tomorrow morning.
Natalie Allen, the CNN anchor, says it will cheer "him" up when Kissinger visits.
Both the interview guest Kissinger and the interview host Allen mention that Kissinger will visit "him". Therefore, the answer is yes, the host is paraphrasing the guest.
Verbatim Quote Guest: "I'm to see him."
Verbatim Quote Host: "him" "have a visit from you"
Classification: Yes.


Given an interview on DATE with the summary: SUMMARY
Guest and Host say the following:
Guest (GUEST-NAME): GUEST-UTTERANCE
Host (HOST-NAME): HOST-UTTERANCE
In the reply, does the host paraphrase something specific the guest says?

Explanation: Let's think step by step.
```



## Requirements

Installed with Python 3.11.7

see `requirements.txt` 

## Questions

Thank you for your comments and questions. You can use GitHub Issues or address me directly (Anna via a.m.wegmann @ uu.nl).

**Correction**: We provide the same running example exchange (QID: CNN-80522-7) for our EMNLP paper, poster and presentation slides. In the Poster/Presentation, we said that this was annotated by 20 annotators, but it was actually annotated by 3. We used dynamic online annotator recruitment. As the first 3 annotators agreed, we did not recruit more for that item.


## Citation

If you use our data or models, consider citing our paper:

```
@article{wegmann2024,
  title={What's Mine becomes Yours: Defining, Annotating and Detecting Context-Dependent Paraphrases in News Interview Dialogs},
  author={Wegmann, Anna and Broek, Tijs van den and Nguyen, Dong},
  journal={arXiv preprint arXiv:2404.06670},
  year={2024}
}
```