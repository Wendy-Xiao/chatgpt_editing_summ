from dis import Instruction
import json
import os
import random
import time
from typing import Any, Dict, List
from rl4lms.envs.text_generation.observation import Observation
from rl4lms.envs.text_generation.reward import RewardFunction,BatchedRewardFunction
from datasets import load_metric
import openai
import pdb
import re
from datetime import datetime
import numpy as np
from torch import device
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
from nltk import sent_tokenize


def extract_keywords(results):
    """
    keyword_list: a list of entities
    keyword_indice_list: a list of indices
    """
    keyword_list = []
    keyword_indice_list = []
    cur_token = ""
    cur_indice = []
    for i, token in enumerate(results):
        if i != 0 and token["start"] == results[i - 1]["end"]:
            if '##' in token["word"]:
                cur_token += token["word"][2:]
            else:
                cur_token += token["word"]
            cur_indice.append(token["index"])
        else:
            if token["score"] > 0.5:
                if token["entity"][0] == "B":
                    if cur_token != "":
                        keyword_list.append(cur_token)
                        keyword_indice_list.append(cur_indice)
                    cur_token = token["word"]
                    cur_indice = [token["index"]]
                else:
                    if i == 0:
                        # error
                        continue
                    elif token["start"] == results[i - 1]["end"] + 1:
                        cur_token += " " + token["word"]
                        cur_indice.append(token["index"])
                    else:
                        # error
                        continue
    keyword_list.append(cur_token)
    keyword_indice_list.append(cur_indice)
    return keyword_list, keyword_indice_list

class GPTSummRewardFunction(RewardFunction):
    def __init__(self, use_all_keywords=True,openai_api='',gpt_engine='',result_dir='/mnt/default/data/cnndm_gptsumm_rl_output/') -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self.openai_api = openai_api
        self.gpt_engine=gpt_engine
        openai.api_key =openai_api #os.getenv("OPENAI_API_KEY")
        self.use_all_keywords = use_all_keywords
        self.result_dir=result_dir

    def generate(self,input_prompt, temperature=0, max_tokens=100,partition_id=None):
        openai.api_key = self.openai_api #os.getenv("OPENAI_API_KEY")
        result = None
        retry_interval_exp=1
        while True:
            try:
                if not partition_id:
                    # Requests will be routed in round robin by default.
                    partition_id = f"sumscience-{datetime.now()}"
                # response=gpt_model.complete(prompt=input_prompt,temperature=temperature,max_tokens=max_tokens)
                response = openai.Completion.create(
                    engine=self.gpt_engine,
                    # engine="text-davinci-003",
                    prompt=input_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=1,
                    top_p=1,
                    frequency_penalty=0.5,
                    presence_penalty=0.0,
                    stop=None,
                    headers={"partition-id": partition_id},
                )
                result = [
                    response["choices"][i]["text"].replace("\n", "").replace(" .", ".").strip()
                    for i in range(len(input_prompt))
                ]
                return result
            except Exception as e:
                # NOTE: openai.error.RateLimitError: Requests to the
                # Deployments_Completion Operation under OpenAI API have
                # exceeded rate limit of your current OpenAI S0 pricing tier.
                # Please retry after 7 seconds. Please contact Azure support
                # service if you would like to further increase the default rate
                # limit.
                if isinstance(e, openai.error.APIConnectionError):
                    # Expontial backoff
                    time.sleep(max(4, 0.5 * (2**retry_interval_exp)))
                    retry_interval_exp += 1
                    print('apiconnection error')
                elif isinstance(e,openai.error.RateLimitError):
                    # error = {"type": type(e).__name__, "message": str(e)}
                    # print(error)
                    # Expontial backoff
                    time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                    retry_interval_exp += 1
                else:
                    # NOTE: openai.error.InvalidRequestError: The response was
                    # filtered due to the prompt triggering Azure OpenAI’s
                    # content management policy.
                    error = {"type": type(e).__name__, "message": str(e)}
                    print(error)
                    return

    def build_prompt_for_correction(
            self,
            all_inputs,
            instructions
        ):
        keyword_list_plus,keyword_list_minus=self.build_keyword_list_from_output(instructions)
        all_prompts = []
        orig_summ=[]
        orig_doc=[]
        new_summ_mapping={}
        for i in range(len(all_inputs)):
            summ_doc = re.findall('<summary>(?P<summ>.*)<document>(?P<doc>.*)',all_inputs[i])
            summ = summ_doc[0][0]
            doc=summ_doc[0][1]
            orig_summ.append(summ)
            orig_doc.append(doc)
            if len(keyword_list_plus[i])==0 and len(keyword_list_minus[i]) == 0:
                continue
            prompt = "Summary: %s \n\nDocument: %s \n\nRewrite the summary for the document," % (
                summ,
                doc,
            )
            if len(keyword_list_plus[i]) > 0:
                if self.use_all_keywords:
                    prompt += " add content related to %s," % (" and ".join(keyword_list_plus[i]))
                else:
                    prompt += " add content related to %s," % (random.choice(keyword_list_plus[i]))
            if len(keyword_list_minus[i]) > 0:
                if self.use_all_keywords:
                    prompt += " delete content related to %s." % (" and ".join(keyword_list_minus[i]))
                else:
                    prompt += " delete content related to %s." % (random.choice(keyword_list_minus[i]))
            prompt += "\n\n New summary: "
            new_summ_mapping[i]=len(all_prompts)
            all_prompts.append(prompt)
        return all_prompts,orig_summ,orig_doc,new_summ_mapping

    def build_keyword_list_from_output(self,instructions):
        plus_list_all=[]
        minus_list_all=[]
        for instruction in instructions:
            plus_text=re.search(r'\<add\>(.*?)\<remove\>',instruction)
            if plus_text:
                plus_list=plus_text.group(1)
                plus_list=plus_list.split(',')
                plus_list=[kw.strip() for kw in plus_list if len(kw.strip())>0 and '##' not in kw ]
            else:
                plus_list=[]

            minus_text=re.search(r'\<remove\>(.*?)(\<remove\>|\<add\>)',instruction+'<remove>')
            if minus_text:
                minus_list=minus_text.group(1)
                minus_list=minus_list.split(',')
                minus_list=[kw.strip() for kw in minus_list if len(kw.strip())>0 and '##' not in kw ]
            else:
                minus_list=[]
            plus_list_all.append(list(set(plus_list)))
            minus_list_all.append(list(set(minus_list)))
        return plus_list_all,minus_list_all

    def __call__(self, prev_observation: Observation,
                action: int,
                current_observation: Observation,
                done: bool,
                meta_info: Dict[str, Any] = None) -> float:
        if done:
            # TBD: considers only one reference for now
            references = [current_observation.target_or_reference_texts]
            instruction_pred = [current_observation.context_text]
            # instruction_pred_prev = [prev_observation.context_text]
            # instruction_pred=instruction_pred_cur+instruction_pred_prev # use the difference between the current and previos as reward
            if len(current_observation.context_text)==0:
                return 0
            all_inputs = [current_observation.prompt_or_input_text]
            # all_inputs_prev = [prev_observation.prompt_or_input_text]
            # all_inputs=all_inputs_cur+all_inputs_prev # use the difference between the current and previos as reward
            prompts,orig_summ,orig_doc,new_summ_mapping= self.build_prompt_for_correction(all_inputs,instruction_pred)
            if len(prompts)==0:
                with open(os.path.join(self.result_dir,'rl_train_%d.jsonl'%(os.getpid())),'a') as of:
                    for i in range(len(all_inputs)):
                        d = {'article':orig_doc[i],'highlights':references[i],'gpt_summ':orig_summ[i],'gpt_summ_orig':orig_summ[i],'updated':False,'rouge_diff':None}
                        json.dump(d,of)
                return 0
            
            predicted = self.generate(prompts)
            all_predictions=[predicted[new_summ_mapping[i]] if i in new_summ_mapping.keys() else orig_summ[i] for i in range(len(all_inputs))]
            predicted=all_predictions
            metric_results_cur = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )
            metric_results_prev = self._metric.compute(
                predictions=orig_summ, references=references, use_stemmer=True
            )
            # reward = metric_results['rouge2'].mid.fmeasure
            rouge_keys = ["rouge1", "rouge2", "rougeL"]
            # scores = [
            #     metric_results[rouge_type].mid.fmeasure for rouge_type in rouge_keys
            # ]

            # use the difference between the current and previos as reward
            scores = [
                metric_results_cur[rouge_type].mid.fmeasure-metric_results_prev[rouge_type].mid.fmeasure for rouge_type in rouge_keys
            ]
            reward = np.mean(scores)*10
            with open(os.path.join(self.result_dir,'rl_train_%d.jsonl'%(os.getpid())),'a') as of:
                for i in range(len(all_inputs)):
                    d = {'article':orig_doc[i],'highlights':references[i],'gpt_summ':all_predictions[i],'gpt_summ_orig':orig_summ[i],'updated':(i in new_summ_mapping.values()),'rouge_diff':scores}
                    json.dump(d,of)
                    of.write('\n')
            return reward
        return 0



class GPTSummBatchedRewardFunction(BatchedRewardFunction):
    def __init__(self, use_all_keywords=True,gpt_engine='',openai_api='',result_dir='/mnt/default/data/cnndm_gptsumm_rl_output/',model=None,reward='kw_match+rouge_diff',kw_model_device=0,beta=1,lambd=2,main_device=0) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self.openai_api=openai_api
        openai.api_key = openai_api #os.getenv("OPENAI_API_KEY")
        self.use_all_keywords = use_all_keywords
        self.result_dir=result_dir
        self.lamb=lambd
        self.beta=beta
        self.compute_time=0
        self.gpt_engine=self.gpt_engine
        self.main_device=main_device
        print('Use model: %s'%(gpt_engine))
        if 'kw_match' in reward:
            ner_tokenizer = AutoTokenizer.from_pretrained(
                "dslim/bert-large-NER",
                cache_dir="/mnt/default/data/pretrained_models" ,
            )
            ner_model = AutoModelForTokenClassification.from_pretrained(
                "dslim/bert-large-NER",
                cache_dir="/mnt/default/data/pretrained_models",
            )
            self.ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer,device=kw_model_device)
        print('alpha: %.2f, beta: %.2f'%(lambd,beta))
    def generate(self,input_prompt, temperature=0, max_tokens=100,partition_id=None):
        openai.api_key = self.openai_api #os.getenv("OPENAI_API_KEY")
        result = None
        retry_interval_exp=1
        while True:
            try:
                if not partition_id:
                    # Requests will be routed in round robin by default.
                    partition_id = f"sumscience-{datetime.now()}"
                # response=gpt_model.complete(prompt=input_prompt,temperature=temperature,max_tokens=max_tokens)
                response = openai.Completion.create(
                    engine=self.gpt_engine,
                    # engine="text-davinci-003",
                    prompt=input_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=1,
                    top_p=1,
                    frequency_penalty=0.5,
                    presence_penalty=0.0,
                    stop=None,
                    headers={"partition-id": partition_id},
                )
                result = [
                    response["choices"][i]["text"].replace("\n", "").replace(" .", ".").strip()
                    for i in range(len(input_prompt))
                ]
                return result
            except Exception as e:
                # NOTE: openai.error.RateLimitError: Requests to the
                # Deployments_Completion Operation under OpenAI API have
                # exceeded rate limit of your current OpenAI S0 pricing tier.
                # Please retry after 7 seconds. Please contact Azure support
                # service if you would like to further increase the default rate
                # limit.
                if isinstance(e, openai.error.APIConnectionError):
                    # Expontial backoff
                    time.sleep(max(4, 0.5 * (2**retry_interval_exp)))
                    retry_interval_exp += 1
                    print('apiconnection error')
                elif isinstance(e,openai.error.RateLimitError):
                    # error = {"type": type(e).__name__, "message": str(e)}
                    # print(error)
                    # Expontial backoff
                    time.sleep(max(4, 0.5 * (2 ** retry_interval_exp)))
                    retry_interval_exp += 1
                else:
                    # NOTE: openai.error.InvalidRequestError: The response was
                    # filtered due to the prompt triggering Azure OpenAI’s
                    # content management policy.
                    error = {"type": type(e).__name__, "message": str(e)}
                    print(error)
                    return

    def build_prompt_for_correction_instruction(
            self,
            all_inputs,
            instructions
    ):
        all_prompts = []
        orig_summ=[]
        orig_doc=[]
        new_summ_mapping={}
        for i in range(len(all_inputs)):
            summ_doc = re.findall(r'Summary: (?P<summ>.*?)\nDocument: (?P<doc>.*?)\n Instruction',all_inputs[i])
            summ = summ_doc[0][0].strip()
            doc=summ_doc[0][1].strip()
            orig_summ.append(summ)
            orig_doc.append(doc)
            if instructions[i]=='':
                continue 
            prompt = "Document: %s \nSummary: %s \nInstructions: %s \nEdit the summary only following the instructions and only output the corrected summary.\nNew summary: " % (
                doc,
                summ,
                instructions[i]
            )
            new_summ_mapping[i]=len(all_prompts)
            all_prompts.append(prompt)
        return all_prompts,orig_summ,orig_doc,new_summ_mapping

    def build_prompt_for_correction(
            self,
            all_inputs,
            # instructions
            keyword_list_plus,keyword_list_minus
        ):
        # keyword_list_plus,keyword_list_minus=self.build_keyword_list_from_output(instructions)
        all_prompts = []
        orig_summ=[]
        orig_doc=[]
        new_summ_mapping={}
        for i in range(len(all_inputs)):
            summ_doc = re.findall('<summary>(?P<summ>.*)<document>(?P<doc>.*)',all_inputs[i])
            summ = summ_doc[0][0]
            doc=summ_doc[0][1]
            orig_summ.append(summ)
            orig_doc.append(doc)
            if len(keyword_list_plus[i])==0 and len(keyword_list_minus[i]) == 0:
                continue
            prompt = "Summary: %s \n\nDocument: %s \n\nRewrite the summary for the document," % (
                summ,
                doc,
            )
            if len(keyword_list_plus[i]) > 0:
                if self.use_all_keywords:
                    prompt += " add content related to %s," % (" and ".join(keyword_list_plus[i]))
                else:
                    prompt += " add content related to %s," % (random.choice(keyword_list_plus[i]))
            if len(keyword_list_minus[i]) > 0:
                if self.use_all_keywords:
                    prompt += " delete content related to %s." % (" and ".join(keyword_list_minus[i]))
                else:
                    prompt += " delete content related to %s." % (random.choice(keyword_list_minus[i]))
            prompt += "\n\n New summary: "
            new_summ_mapping[i]=len(all_prompts)
            all_prompts.append(prompt)
        return all_prompts,orig_summ,orig_doc,new_summ_mapping

    def build_keyword_list_from_output(self,instructions):
        plus_list_all=[]
        minus_list_all=[]
        for instruction in instructions:
            plus_text=re.search(r'\<add\>(.*?)\<remove\>',instruction)
            if plus_text:
                plus_list=plus_text.group(1)
                plus_list=plus_list.split(',')
                plus_list=[kw.strip() for kw in plus_list if len(kw.strip())>1 and '##' not in kw ]
            else:
                plus_list=[]

            minus_text=re.search(r'\<remove\>(.*?)(\<remove\>|\<add\>)',instruction+'<remove>')
            if minus_text:
                minus_list=minus_text.group(1)
                minus_list=minus_list.split(',')
                minus_list=[kw.strip() for kw in minus_list if len(kw.strip())>1 and '##' not in kw ]
            else:
                minus_list=[]
            plus_list_all.append(list(set(plus_list)))
            minus_list_all.append(list(set(minus_list)))
        return plus_list_all,minus_list_all

    
    def compute_reward_kw_match(self,predicted_summary_batch,gt_summary_batch,original_summary_batch):
        gt_summary_batch = [s[0] for s in gt_summary_batch]
        all_summ = predicted_summary_batch+gt_summary_batch+original_summary_batch
        batch_size = len(predicted_summary_batch)
        all_summ_results = self.ner_pipeline(all_summ)
        gen_summ_results = all_summ_results[:batch_size]
        gt_summ_results = all_summ_results[batch_size:2*batch_size]
        prev_summ_results = all_summ_results[2*batch_size:]
        assert len(gt_summ_results) == len(prev_summ_results)
        f1_diff=[]
        for i_batch in range(len(gt_summ_results)):
            keyword_list_gt, _ = extract_keywords(gt_summ_results[i_batch])
            keyword_list_gen, _ = extract_keywords(gen_summ_results[i_batch])
            keyword_list_prev, _ = extract_keywords(prev_summ_results[i_batch])
            num_overlap=len(set(keyword_list_gt).intersection(set(keyword_list_gen)))
            r = num_overlap/(len(keyword_list_gt)+1e-9)
            p=num_overlap/(len(keyword_list_gen)+1e-9)
            f_gen=2*r*p/(r+p+1e-9)

            num_overlap=len(set(keyword_list_gt).intersection(set(keyword_list_prev)))
            r = num_overlap/(len(keyword_list_gt)+1e-9)
            p=num_overlap/(len(keyword_list_prev)+1e-9)
            f_prev=2*r*p/(r+p+1e-9)
            f = f_gen-f_prev
            # if len(keyword_list_gen)==0 and len(keyword_list_gt)==0:
            #     r=1
            #     p=1
            #     f=1
            f1_diff.append(f)
        return f1_diff

    def __call__(self, prompt_texts: List[str],
                 gen_texts: List[str],
                 ref_texts: List[List[str]],
                 dones: List[bool],
                 meta_infos: List[Dict[str, Any]] = None) -> float:
        scores = [0 for i in range(len(dones))]
        done_idx=[i for i,d in enumerate(dones) if d]
        if len(done_idx)==0:
            return scores
        self.compute_time+=1
        references=[ref_texts[i] for i in done_idx]
        instruction_pred=[gen_texts[i] for i in done_idx]
        all_inputs = [prompt_texts[i] for i in done_idx]
        if 'keyword_list_plus' in meta_infos[0]:
            t = 'kw'
            #CNNDM Dataset
            all_kw_list_plus=[meta_infos[i]['keyword_list_plus'] for i in done_idx]
            all_kw_list_minus=[meta_infos[i]['keyword_list_minus'] for i in done_idx]
            #build prompts to generate summaries
            keyword_list_plus,keyword_list_minus=self.build_keyword_list_from_output(instruction_pred)
            prompts,orig_summ,orig_doc,new_summ_mapping= self.build_prompt_for_correction(all_inputs,keyword_list_plus,keyword_list_minus)
        elif 'instruction' in meta_infos[0]:
            t='instruction'
            # DeFacto Dataset
            instruction_gt = [meta_infos[i]['instruction'] for i in done_idx]
            #build prompts to generate summaries
            prompts,orig_summ,orig_doc,new_summ_mapping=self.build_prompt_for_correction_instruction(all_inputs,instruction_pred)
        

        if len(prompts)==0:
            final_scores = [0 for i in range(len(orig_summ))]
        else:
            predicted = self.generate(prompts)
            if t=='instruction':
                predicted=[sent_tokenize(s)[0] for s in predicted]
            all_predictions=[predicted[new_summ_mapping[i]] if i in new_summ_mapping.keys() else orig_summ[i] for i in range(len(all_inputs))]
            # predicted=all_predictions
            metric_results_cur = self._metric.compute(
                predictions=all_predictions, references=references, use_stemmer=True,use_aggregator=False
            )
            metric_results_prev = self._metric.compute(
                predictions=orig_summ, references=references, use_stemmer=True,use_aggregator=False
            )
            
            # use the difference between the current and previos as reward
            rouge_keys = ["rouge1", "rouge2", "rougeL"]
            scores_done = [np.mean([
                metric_results_cur[rouge_type][i].fmeasure-metric_results_prev[rouge_type][i].fmeasure for rouge_type in rouge_keys
            ]) for i in range(len(orig_summ))]

            #kw match reward
            # if t=='kw':
            kw_match_diff=self.compute_reward_kw_match(all_predictions,references,orig_summ)
            final_scores=[scores_done[i]*self.lamb+self.beta*kw_match_diff[i] for i in range(len(scores_done))]
        if t=='instruction':
            for i in range(len(instruction_pred)):
                if instruction_pred[i]=='' and instruction_gt[i]!='':
                    final_scores[i]=-10 #penalty
                if instruction_pred[i]=='' and instruction_gt[i]=='':
                    final_scores[i]=10 #reward
                if instruction_pred[i]!='' and instruction_gt[i]=='':
                    final_scores[i]=-10 #penalty
                    
            # elif t=='instruction':

            
            # with open(os.path.join(self.result_dir,'rl_train_%d_%s_%d.jsonl'%(os.getpid(),str(datetime.now()),random.randint(1000,9999))),'a') as of:
            #     for i in range(len(all_inputs)):
            #         d = {'article':orig_doc[i],
            #              'highlights':references[i],
            #              'gpt_summ':all_predictions[i],
            #              'gpt_summ_orig':orig_summ[i],
            #              'updated':(i in new_summ_mapping.values()),
            #              'rouge_diff':scores_done[i]/10,
            #              'rouge_instruction':instruction_scores[i],
            #              'final_score':final_scores[i]}
            #         json.dump(d,of)
            #         of.write('\n')
        for i,idx in enumerate(done_idx):
            scores[idx] = final_scores[i]
        torch.cuda.set_device(self.main_device)
        return scores


