"""
    Test evaluation of in-context learnign responses
"""
import json
import os
from unittest import TestCase
import evaluate_icl_responses
from build_prompts import INSTRUCT_PROMPT_TEST
from paraphrase.utility.project_functions import get_dir_to_src


class Test(TestCase):
    def test_main(self):
        response_path = "fixtures/in-context/Responses-openchat-3.5-0106_4bit_5-fixture-prompts.json"
        prompt_path = "fixtures/in-context/5-fixture-prompts.json"
        evaluate_icl_responses.main(response_path, prompt_path)

    def test_eval_instruction(self):
        # DEV
        prompt_path = "../../result/Models/0-8-dev_prompts.json"
        # response_path = "../result/Models/Responses-openchat-3.5-01064bit_0-8-dev_prompts_20240228_151136.json"
        # response_path = "../result/Models/Responses-gemma-7b-it4bit_0-8-dev_prompts_20240229_122355.json"
        # response_path = "../result/Models/Responses-Llama-2-7b-hf4bit_0-8-dev_prompts_20240229_110426.json"
        # response_path = "../result/Models/Responses-Mistral-7B-Instruct-v0.24bit_0-8-dev_prompts_20240229_051126.json"
        # response_path = "../result/Models/Responses-Mistral-7B-v0.14bit_0-8-dev_prompts_20240228_183758.json"
        # response_path = "../result/Models/Responses-Mixtral-8x7B-Instruct-v0.14bit_0-8-dev_prompts_20240229_015248.json"
        # response_path = "../result/Models/Responses-vicuna-7b-v1.54bit_0-8-dev_prompts_20240229_071500.json"

        # response_path = "../result/Models/Responses-Llama-2-70b-hf4bit_0-8-dev_prompts_20240302_080659.json"
        # response_path = "../result/Models/Responses-openchat-3.5-0106default_0-8-dev_prompts_20240302_114320.json"
        # response_path = "../result/Models/Responses-Mistral-7B-v0.1default_0-8-dev_prompts_20240302_135429.json"

        # evaluate_prompt.main(response_path, prompt_path)

        # GPT-4
        # response_path = "../result/Models/Responses-GPT4_0-4-dev_prompts_20240301_205803.json"
        # evaluate_prompt.main(response_path, prompt_path, gpt4=True)

        # test path
        response_path = "../../result/Models/RMV_Responses-Llama-2-7b-hfdefault_0-8-test_prompts_20240307_162940.json"
        # response_path = "../../result/Models/RMV_Responses-vicuna-7b-v1.5default_0-8-test_prompts_20240307_054555.json"
        # response_path = "../../result/Models/RMV_Responses-Mistral-7B-Instruct-v0.2default_0-8-test_prompts_20240307_040712.json"
        # response_path = "../../result/Models/RMV_Responses-openchat-3.5-0106default_0-8-test_prompts_20240307_015142.json"
        # response_path = "../../result/Models/RMV_Responses-gemma-7b-itdefault_0-8-test_prompts_20240307_065354.json"


        # response_path = "../../result/Models/Responses-Llama-2-70b-hf4bit_0-8-test_prompts_20240319_215135.json"
        response_path = "../../result/Models/RMV_COMB_Responses-Llama-2-70b-hfdefault_0-8-test_prompts_20240325_111822.json"
        response_path = "../../result/Models/RMV_Responses-Mixtral-8x7B-Instruct-v0.1default_0-8-test_prompts_20240318_113504.json"
        response_path = "../../result/Models/RMV_Responses-gemma-7b-itdefault_0-8-test_prompts_20240307_065354.json"
        response_path = "../../result/Models/RMV_Responses-openchat-3.5-0106default_0-8-test_prompts_20240307_015142.json"
        response_path = "../../result/Models/RMV_Responses-Mistral-7B-Instruct-v0.2default_0-8-test_prompts_20240307_040712.json"
        response_path = "../../result/Models/RMV_Responses-vicuna-7b-v1.5default_0-8-test_prompts_20240307_054555.json"
        response_path = "../../result/Models/RMV_Responses-Llama-2-7b-hfdefault_0-8-test_prompts_20240307_162940.json"
        prompt_path = INSTRUCT_PROMPT_TEST
        evaluate_icl_responses.main(response_path, prompt_path, no_prepended_prompt=True)
        response_path = "../../result/Models/Responses-GPT4_0-8-test_prompts_20240307_140445.json"
        # evaluate_icl_responses.main(response_path, prompt_path, no_prepended_prompt=True)

    def test_remove_prompt(self):
        response_path = get_dir_to_src() + "/../result/Models/Responses-Mixtral-8x7B-Instruct-v0.1default_0-8-test_prompts_20240318_113504.json"
        # response_path = get_dir_to_src() + "/../result/Models/Responses-Llama-2-7b-hfdefault_0-8-test_prompts_20240307_162940.json"
        # response_path = get_dir_to_src() + "/../result/Models/Responses-vicuna-7b-v1.5default_0-8-test_prompts_20240307_054555.json"
        # response_path = get_dir_to_src() + "/../result/Models/Responses-Mistral-7B-Instruct-v0.2default_0-8-test_prompts_20240307_040712.json"
        # response_path = get_dir_to_src() + "/../result/Models/Responses-openchat-3.5-0106default_0-8-test_prompts_20240307_015142.json"
        # response_path = get_dir_to_src() + "/../result/Models/Responses-gemma-7b-itdefault_0-8-test_prompts_20240307_065354.json"
        response_path = get_dir_to_src() + "/../result/Models/COMB_Responses-Llama-2-70b-hfdefault_0-8-test_prompts_20240325_111822.json"
        prompt_path = INSTRUCT_PROMPT_TEST
        response_path = evaluate_icl_responses.remove_prompt_from_reply(response_path, prompt_path)

    def test_combine_json(self):
        path_1 = "../../result/Models/Responses-Llama-2-70b-hfdefault_0-8-test_prompts_20240325_111822.json"
        path_2 = "../../result/Models/Responses-Llama-2-70b-hfdefault_0-8-test_prompts_20240326_172622.json"
        path_3 = "../../result/Models/Responses-Llama-2-70b-hfdefault_0-8-test_prompts_20240327_033749.json"

        model_responses_1 = evaluate_icl_responses.load_json_file(path_1)
        model_responses_2 = evaluate_icl_responses.load_json_file(path_2)
        model_responses_3 = evaluate_icl_responses.load_json_file(path_3)

        for q_id, response_list in model_responses_1.items():
            response_list.extend(model_responses_2[q_id])
            response_list.extend(model_responses_3[q_id])

        filename = os.path.basename(path_1)
        filename = f"{os.path.dirname(path_1)}/COMB_{filename}"

        with open(filename, "w") as file:
            json.dump(model_responses_1, file, indent=4)


