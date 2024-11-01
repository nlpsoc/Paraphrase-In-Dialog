"""
    Tests building a prompt from the template
        also returns the prompt for a specific given ID
"""
from unittest import TestCase

from paraphrase.interview_data import MediaSumProcessor
from paraphrase.prompt_templates import FS_PROMPT_TEMPLATES, build_few_shot_prompt, FEW_SHOT_EXAMPLES


class Test(TestCase):
    def test_build_one_prompt(self):
        q_id = "CNN-80522-7"  # figure 1 prompt
        instruction_prompt_template = FS_PROMPT_TEMPLATES[0]  # prompt template that is close to instruction

        # get the q-item for the question in question
        interview = MediaSumProcessor()
        q_item = interview.get_qdict_from_qid(q_id)

        prompt = build_few_shot_prompt(instruction_prompt_template, FEW_SHOT_EXAMPLES, q_item)
        print(prompt)
