from datetime import datetime
import json
import random
import re
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm
import evaluate
from nltk import sent_tokenize
import numpy as np


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
            if token["score"] > 0.8:
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

def compute_knowledge_f1(nlp, gt_summary_batch, predicted_summary_batch):
    all_gt_summ = [s for s in gt_summary_batch]
    all_gen_summ = [s for s in predicted_summary_batch]
    gt_summ_results = nlp(all_gt_summ)
    gen_summ_results = nlp(all_gen_summ)
    recall=[]
    precision=[]
    f1=[]
    for i_batch in range(len(gt_summ_results)):
        keyword_list_gt, _ = extract_keywords(gt_summ_results[i_batch])
        keyword_list_gen, _ = extract_keywords(gen_summ_results[i_batch])
        num_overlap=len(set(keyword_list_gt).intersection(set(keyword_list_gen)))
        r = num_overlap/(len(set(keyword_list_gt))+1e-9)
        p=num_overlap/(len(set(keyword_list_gen))+1e-9)
        f=2*r*p/(r+p+1e-9)
        if len(keyword_list_gen)==0 and len(keyword_list_gt)==0:
            r=1
            p=1
            f=1
        recall.append(r)
        precision.append(p)
        f1.append(f)
    return recall,precision,f1

def compute_knowledge_f1_all(nlp, all_gt_summ,all_predicted_summ):
    recall=[]
    precision=[]
    f1=[]
    for i in tqdm(range(0,len(all_gt_summ),32)):
        gt_summary_batch=all_gt_summ[i:i+32]
        predicted_summary_batch=all_predicted_summ[i:i+32]
        r,p,f=compute_knowledge_f1(nlp, gt_summary_batch, predicted_summary_batch)
        recall.extend(r)
        precision.extend(p)
        f1.extend(f)
    return recall,precision,f1


if __name__=='__main__':
    summary_file=''
    cache_dir='./data/pretrained_models'
    data_file = './data/cnndm_gptsumm_chatgpt/cnndm_chatgptsumm_with_kw_test.jsonl' 
    all_data=[]
    with open(data_file,'r') as of:
        all_lines=of.readlines()
        for i,l in enumerate(all_lines):
            all_data.append(json.loads(l))
    rouge = evaluate.load('rouge')

    ner_tokenizer = AutoTokenizer.from_pretrained(
        "dslim/bert-large-NER",
        cache_dir=cache_dir ,
    )
    ner_model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-large-NER",
        cache_dir=cache_dir,
    )
    nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=7)
    all_gpt_summ={}
    with open(summary_file,'r') as of:
        all_lines=of.readlines()
        for l in all_lines:
            d = json.loads(l)
            all_gpt_summ[d['id']]=d['gpt_summ']
    all_predicted_summ=[]       
    for d in all_data:
        all_predicted_summ.append(all_gpt_summ[d['id']])

    all_reference_summ=[d['highlights'] for d in all_data]
    predicted = [' '.join(sent_tokenize(s)[:3]) for s in all_predicted_summ]
    recall,precision,f1=compute_knowledge_f1_all(nlp,all_reference_summ,predicted)
    print('knowledge recall: %.2f, knowledge precision: %.2f, knowledge f1: %.2f.'%(np.mean(recall)*100,np.mean(precision)*100,np.mean(f1)*100))
    metric_results = rouge.compute(
        predictions=predicted, references=all_reference_summ, use_stemmer=True
    )
    print('ROUGE 1: %.4f, ROUGE 2: %.4f, ROUGE L: %.4f, ROUGE LSum: %.4f'%(metric_results['rouge1'],metric_results['rouge2'],metric_results['rougeL'],metric_results['rougeLsum']))