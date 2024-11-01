"""
    Test in-context learning
"""
import os
from unittest import TestCase

from build_prompts import INSTRUCT_PROMPT_DEV, INSTRUCT_PROMPT_TEST
from paraphrase.utility.project_functions import get_dir_to_src
from incontext_prompting import main


class Test(TestCase):

    def test_incontext_gpt4(self):
        """
            ATTENTION: this costs moneys if the API is set
        :return:
        """
        output_dir = get_dir_to_src() + "/../output/gpt4/"
        assert os.path.isdir(output_dir), f"output_dir: {output_dir} does not exist."
        # on dev set
        # main(prompt_path=INSTRUCT_PROMPT_DEV, gpt4=True, self_consistency=3, output_path=output_dir)

        # on test set
        main(prompt_path=INSTRUCT_PROMPT_TEST, gpt4=True, self_consistency=3, output_path=output_dir)

    def test_incontext_llm(self):
        main(prompt_path=INSTRUCT_PROMPT_DEV, gpt4=False, huggingface_id="openai-community/gpt2", load_in=None)
