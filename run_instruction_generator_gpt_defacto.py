
import evaluate
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from dataclasses import dataclass, field, fields
from typing import Optional
import os
import json
import numpy as np
from enum import Enum

import nltk
from tqdm import tqdm
import os
from nltk import tokenize
from nltk.corpus import stopwords
import string
import json
import pdb
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["WANDB_DISABLED"] = "true"

STOPWORDS = list(set(stopwords.words("english")))
PUNCTUATIONS = string.punctuation

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="instruction_generation", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use"},
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path of the dataset to use."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict(
            (field.name, getattr(self, field.name))
            for field in fields(self)
            if field.init
        )

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    load_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the trained models"},
    )
    checkpoint_num: Optional[int] = field(
        default=5000,
        metadata={"help": "the step number of the checkpoint to load"},
    )
    ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to the model to load"},
    )
    pretrained_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to the model to load"},
    )

    outfile_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of output file"},
    )


def make_new_model(model, tokenizer):
    # tokenizer.add_special_tokens(
    #     {"additional_special_tokens": ["<summary>", "<document>"]}
    # )
    tokenizer.add_tokens(["<summary>", "<document>"], special_tokens=True)
    tokenizer.add_tokens(["<add>", "<remove>","<placeholder>"], special_tokens=False)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def extract_keywords_ngram(document, n=[2]):
    """
    keyword_list: a list of entities
    """
    keyword_list = []
    tokens = tokenize.word_tokenize(document)
    for i in n:
        keyword_list.extend(list(nltk.ngrams(tokens, i)))
    final_keyword_list = []
    for kw in keyword_list:
        has_punctuation = any([(w in PUNCTUATIONS) for w in kw])
        has_stop = any([(w in STOPWORDS) for w in kw])
        if not (has_punctuation or has_stop):
            final_keyword_list.append(" ".join(kw))
    final_keyword_list = list(set(final_keyword_list))
    return final_keyword_list




def preprocess_instruction(example, tokenizer):

    input_str = 'Generate instruction to correct the summary for the given document. \nSummary: %s \nDocument: %s'%(example["gpt_summ"],example["document"])
    output_str=example['instruction']
    labels = tokenizer(output_str, truncation=True)["input_ids"]
    model_input = tokenizer(input_str, truncation=True)["input_ids"]
    example["input_ids"] = model_input
    example["labels"] = labels
    example["document_id"] = example["id"]
    return example



def compute_metrics(eval_preds):
    metric = evaluate.load("rouge")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # print(labels)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    # pdb.set_trace()
    return result


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args_json = training_args.to_json_string()
    data_args_json = data_args.to_json_string()
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    with open(os.path.join(training_args.output_dir, "train_args.json"), "w") as of:
        json.dump(training_args_json, of)
    with open(os.path.join(training_args.output_dir, "data_args.json"), "w") as of:
        json.dump(data_args_json, of)
    if model_args.load_from_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(training_args.output_dir, "instruction_generator_tokenizer")
        )
        if model_args.ckpt_path:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.ckpt_path
        )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                os.path.join(
                    training_args.output_dir, "checkpoint-%d" % (model_args.checkpoint_num)
                )
            )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            cache_dir="%s/pretrained_models" % (model_args.pretrained_cache_path),
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            cache_dir="%s/pretrained_models" % (model_args.pretrained_cache_path),
        )
        # model.parallelize()
        model, tokenizer = make_new_model(model, tokenizer)
        tokenizer.save_pretrained(
            os.path.join(training_args.output_dir, "instruction_generator_tokenizer")
        )

    if training_args.do_train:
        train_dataset=[]
        with open(os.path.join(data_args.dataset_path,'train.jsonl'),'r') as of:
            all_lines=of.readlines()
            for i,line in enumerate(all_lines):
                train_dataset.append(json.loads(line))
                # if i>200:
                #     break
        train_dataset_dict={'document':[train_dataset[i]['article'] for i in range(len(train_dataset))],
                            'instruction': [train_dataset[i]['feedback']['instruction'] for i in range(len(train_dataset))],
                            'id': [train_dataset[i]['doc_id'] for i in range(len(train_dataset))],
                            'gpt_summ': [train_dataset[i]['candidate'] for i in range(len(train_dataset))],
                            }
        train_dataset=Dataset.from_dict(train_dataset_dict)
        train_dataset = train_dataset.map(
            preprocess_instruction,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=train_dataset.column_names,
        )

        val_dataset=[]
        with open(os.path.join(data_args.dataset_path,'val.jsonl'),'r') as of:
            all_lines=of.readlines()
            for i,line in enumerate(all_lines):
                val_dataset.append(json.loads(line))

        val_dataset_dict={'document':[val_dataset[i]['article'] for i in range(len(val_dataset))],
                            'instruction': [val_dataset[i]['feedback']['instruction'] for i in range(len(val_dataset))],
                            'id': [val_dataset[i]['doc_id'] for i in range(len(val_dataset))],
                            'gpt_summ': [val_dataset[i]['candidate'] for i in range(len(val_dataset))],
                            }
        val_dataset=Dataset.from_dict(val_dataset_dict)
        val_dataset = val_dataset.map(
            preprocess_instruction,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=val_dataset.column_names,
        )
    if training_args.do_predict:
        training_args.do_eval=False
        test_dataset=[]
        with open(os.path.join(data_args.dataset_path,'test.jsonl'),'r') as of:
            all_lines=of.readlines()
            for line in all_lines:
                test_dataset.append(json.loads(line))
        test_dataset = [d for d in test_dataset if len(d['feedback']['summary'])!=0]
        test_dataset_dict={'document':[test_dataset[i]['article'] for i in range(len(test_dataset))],
                            'instruction': [test_dataset[i]['feedback']['instruction'] for i in range(len(test_dataset))],
                            'id': [test_dataset[i]['doc_id'] for i in range(len(test_dataset))],
                            'gpt_summ': [test_dataset[i]['candidate'] for i in range(len(test_dataset))],
                            }
        test_dataset=Dataset.from_dict(test_dataset_dict)
        test_dataset = test_dataset.map(
            preprocess_instruction,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=test_dataset.column_names,
        )
    # pdb.set_trace()
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=val_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )
    # trainer.is_model_parallel=True
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            print("load from %s" % (checkpoint))
        trainer.train(resume_from_checkpoint=checkpoint)
    if training_args.do_predict:
        predict_results = trainer.predict(test_dataset)
        decoded_preds = tokenizer.batch_decode(
            predict_results[0], skip_special_tokens=True
        )
        print(predict_results[-1])
        if model_args.outfile_name:
            summary_file = os.path.join(training_args.output_dir, "generated_instructions_%s.jsonl"%(model_args.outfile_name))
        else:
            summary_file = os.path.join(training_args.output_dir, "generated_instructions.jsonl")
        with open(summary_file,'w') as of:
            for i,s in enumerate(decoded_preds):
                result = {'id':test_dataset["document_id"][i],'gen_instruction':s}
                json.dump(result,of)
                of.write('\n')
