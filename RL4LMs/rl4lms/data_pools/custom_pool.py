from datetime import datetime
import json
import os
import random
import time
from rl4lms.data_pools.custom_text_generation_pools import CommonGen
from rl4lms.data_pools.text_generation_pool import TextGenPool, Sample
from datasets import load_from_disk,load_dataset
from tqdm import tqdm

     
class InstructionData(TextGenPool):
    
    @classmethod
    def prepare(cls, split: str, data_path='/mnt/default/data/cnndm_gptsumm/',data_name='cnndm_gptsumm_with_kw',subset=True,multi_iter=False):
        dataset=[]
        # with open(os.path.join(data_path,'cnndm_gptsumm_with_kw_train.jsonl'),'r') as of:
        file_name=os.path.join(data_path,'%s_%s.jsonl'%(data_name,split))
        if subset and split=='train':
            file_name=os.path.join(data_path,'%s_%s_subset.jsonl'%(data_name,split))
        if multi_iter and split == 'test':
            file_name = os.path.join(data_path,'%s_%s.jsonl'%(data_name,split))
        elif multi_iter:
            file_name = os.path.join(data_path,'%s_%s_subset.jsonl'%(data_name,split))
        print('load data from %s'%(file_name))
        with open(file_name,'r') as of:
            all_lines=of.readlines()
            for i,line in enumerate(all_lines):
                dataset.append(json.loads(line))
                # if i>50:
                #     break
                # if split=='val' and i>1000:
                #     break
                # if split =='test' and i>10:
                #     break
        # if split=='val':
        #     dataset=random.choices(dataset,k=1000)
        if split=='test':
            dataset=random.choices(dataset,k=10)
            
        samples = []
        for ix, item in tqdm(enumerate(dataset),
                                desc="Tokenizing dataset",
                                total=len(dataset)):
            # gen_summ=generate([item['article']+'\n\nTL;DR: '])
            input_str = (
                    " <summary> "
                    + item['gpt_summ']
                    + " <document> "
                    + item["article"]
                )
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text= input_str,
                            references=[item["highlights"]],
                            meta_data={'keyword_list_plus':item['keyword_list_plus'],
                            'keyword_list_minus' : item['keyword_list_minus']}
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance


             
class DefactoData(TextGenPool):
    
    @classmethod
    def prepare(cls, split: str, data_path='/mnt/default/gpt_iter_summ/DeFacto/data',debug=False,with_empty=False):
        dataset=[]
        print('load data from %s'%(split))
        # with open(os.path.join(data_path,'cnndm_gptsumm_with_kw_train.jsonl'),'r') as of:
        file_name=os.path.join(data_path,'%s.jsonl'%(split))
        with open(file_name,'r') as of:
            all_lines=of.readlines()
            for i,line in enumerate(all_lines):
                dataset.append(json.loads(line))
                if debug and i>50:
                    break
                if split =='test' and i>10:
                    break
        samples = []
        if not with_empty:
            dataset = [item for item in dataset if item['feedback']['summary'] is not None and len(item['feedback']['summary'])!=0]
        for ix, item in tqdm(enumerate(dataset),
                                desc="Tokenizing dataset",
                                total=len(dataset)):
            if item['feedback']['summary'] is not None and len(item['feedback']['summary'])!=0:
                references = [item['feedback']['summary']]
            else:
                references = [item['candidate']]
            # gen_summ=generate([item['article']+'\n\nTL;DR: '])
            input_str = 'Generate instruction to correct the summary for the given document. \nSummary: %s \nDocument: %s\n Instruction: '%(item['candidate'],item['article'])
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text= input_str,
                            references=references,
                            meta_data={'instruction':item['feedback']['instruction']}
                            )
            samples.append(sample)

        pool_instance = cls(samples)
        return pool_instance
