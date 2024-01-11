"""
Create a huggingface dataset from role annotations

Code inspiration drawn from:
dataset creation - https://gist.github.com/jangedoo/7ac6fdc7deadc87fd1a1124c9d4ccce9
evaluation - https://wandb.ai/biased-ai/Named-Entity%20Recognition%20on%20HuggingFace/reports/Named-Entity-Recognition-on-HuggingFace--Vmlldzo3NTk4NjY
https://sanjayasubedi.com.np/deeplearning/training-ner-with-huggingface-transformer/

# try roberta large
# 1) fiddle a little with learning rate of model
# 2) annotate more data, try a larger model
# 3) adaptation (definitely will bump it a few points, low-risk) but if it's something systematic then this won't fix it
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, Features, Value, ClassLabel, Sequence, load_metric, DatasetDict
import json
import torch
import numpy as np
import random
# from sklearn.model_selection import KFold
import os

def get_tokens_with_entities(seed=123):
    with open('reformat_annotated_examples.json', 'r') as infile:
        d = json.load(infile)
    all_tokens = []
    all_entities = []
    fname = 'd_keys_order.txt'
    if os.path.isfile(fname): 
        d_keys = []
        with open(fname, 'r') as infile: 
            for line in infile: 
                d_keys.append(line.strip())
    else: 
        d_keys = list(sorted(d.keys()))
        random.seed(seed)
        random.shuffle(d_keys)
        with open(fname, 'w') as outfile: 
            for orig_name in d_keys: 
                outfile.write(orig_name + '\n')
    for i, orig_name in enumerate(d_keys):
        tokens = d[orig_name]['tokens'] # ['I', 'am', 'a', 'chemist', '.']
        entities = d[orig_name]['entities'] # ['O', 'O', 'O', 'B-R', 'O']
        all_tokens.append(tokens)
        all_entities.append(entities)
    return all_tokens, all_entities

def tokenize_and_align_labels(examples):
    '''
    from
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb
    '''
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    label_all_tokens = True

    labels = []
    for i, label in enumerate(examples["role_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    '''
    from
    https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb
    '''
    metric = load_metric("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = ['O', 'B-R']

    label_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=label_predictions, references=true_labels)
    return {
        "precision": results["R"]['precision'],
        "recall": results["R"]['recall'],
        "f1": results["R"]['f1'],
        #"accuracy": results["overall_accuracy"],
    }

def create_dataset(seed=123):
    # train and then test on the train and make sure it overfits
    tokens, entities = get_tokens_with_entities(seed=seed)

    ids = range(len(tokens))

    data = {
        "id": ids,
        "role_tags": entities,
        "tokens": tokens
    }
    features = Features({
        "tokens": Sequence(Value("string")),
        "role_tags": Sequence(ClassLabel(names=['O', 'B-R'])),
        "id": Value("int32")
    })
    ds = Dataset.from_dict(data, features)
    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True)
    
    # split into train / test / val
    train_testval = tokenized_ds.train_test_split(test_size=400, shuffle=False)
    testval = train_testval['test'].train_test_split(test_size=200, shuffle=False)
    tokenized_ds = DatasetDict({
        'train': train_testval['train'],
        'test': testval['test'],
        'val': testval['train']})

    return tokenized_ds

class NpEncoder(json.JSONEncoder):
    '''
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def get_model(model_type): 
    if model_type in set(['roberta_1ep', 'roberta_10ep']):
        model_path = os.path.join('/home/lucyl/llm_social_identities/outputs/identity/', model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        model = AutoModelForTokenClassification.from_pretrained(model_type, num_labels=2)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return tokenizer, model, data_collator

def train_model(tokenized_ds, model_type, ignore_done=True):
#     kf = KFold(n_splits=6)
#     kf.get_n_splits(tokenized_ds)

    lrs = [3e-5, 2e-5, 1e-5] # default is usually 2e-5

    for lr in lrs:
#         for fold_num, (train_idx, val_idx) in enumerate(kf.split(tokenized_ds)):
        tokenizer, model, data_collator = get_model(model_type)

#         output_dir = f"./{model_type}-roles-{lr}-{fold_num}"
        output_dir = f"./{model_type}-roles-{lr}"
        print('**********************', output_dir, '**********************')
        if ignore_done and os.path.exists(os.path.join(output_dir, 'results.json')):
            print("EXPERIMENT ALREADY EXISTS")
            continue

        training_args = TrainingArguments(
            output_dir,
            evaluation_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            weight_decay=0.01,
            save_total_limit=1,
            save_strategy='epoch',
            metric_for_best_model='eval_f1',
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['val'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.evaluate()

        predictions, labels, _ = trainer.predict(tokenized_ds['test'])
        predictions = np.argmax(predictions, axis=2)

        label_list = ['O', 'B-R']

        label_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric = load_metric("seqeval")
        results = metric.compute(predictions=label_predictions, references=true_labels)
        with open(os.path.join(output_dir, 'token_test_results.json'), 'w') as outfile:
            json.dump(results, outfile, cls=NpEncoder)
            
        label_predictions, true_labels = get_word_level_pred(predictions, labels)
        results = metric.compute(predictions=label_predictions, references=true_labels)
        with open(os.path.join(output_dir, 'word_test_results.json'), 'w') as outfile:
            json.dump(results, outfile, cls=NpEncoder)
            
def get_word_level_pred(predictions, labels): 
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    label_predictions = []
    true_labels = []
    for i, prediction in enumerate(predictions): 
        label = labels[i]
        example = tokenized_ds['test'][i]
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        word_ids = tokenized_inputs.word_ids(batch_index=0)

        word_pred = set([word_ids[j] for j, lab in enumerate(prediction) if (lab == 1 and label[j] != -100)])
        word_true = set([word_ids[j] for j, lab in enumerate(label) if (lab == 1 and label[j] != -100)])

        word_pred_lab = ['B-R' if word_idx in word_pred else 'O' for word_idx in range(len(example["tokens"]))]
        word_true_lab = ['B-R' if word_idx in word_true else 'O' for word_idx in range(len(example["tokens"]))]

        label_predictions.append(word_pred_lab)
        true_labels.append(word_true_lab)
    return label_predictions, true_labels
            
def word_level_eval_debug(tokenized_ds): 
    '''
    Inspect model as if we had just trained it
    (This is mostly for debugging)
    '''
    child_dir = next(os.walk('.'))[1]
    child_dir = [di for di in child_dir if di.startswith('roberta_')]
    for curr_dir in child_dir: 
        
        model_path = ''
        for f in os.listdir(curr_dir): 
            if f.startswith('checkpoint'): 
                model_path = os.path.join(curr_dir, f)
                break

        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        trainer = Trainer(
            model=model,
            train_dataset=tokenized_ds['train'],
            eval_dataset=tokenized_ds['val'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    
        predictions, labels, _ = trainer.predict(tokenized_ds['test'])

        predictions = np.argmax(predictions, axis=2)
        
        label_predictions, true_labels = get_word_level_pred(predictions, labels)

        metric = load_metric("seqeval")
        results = metric.compute(predictions=label_predictions, references=true_labels)
        print(results)
        break
#         with open(os.path.join(curr_dir, 'word_level_results.json'), 'w') as outfile:
#             json.dump(results, outfile, cls=NpEncoder)

if __name__ == "__main__":
    tokenized_ds = create_dataset()
    
    for model_type in ['roberta_10ep', 'roberta-base', 'roberta_1ep', 'roberta-large']:
        train_model(tokenized_ds, model_type)
