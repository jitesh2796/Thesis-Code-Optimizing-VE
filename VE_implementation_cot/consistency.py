### Module 0: Importing Libraries
import pandas as pd
import numpy as np
import pprint
import os 
from time import time 
from dotenv import load_dotenv
import json
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI
import certifi

## Importing VE libraries
from utils import *
from dataset_utils import read_wikiqa_data
from prompt_helper import get_joint_prompt_helper, normalize_prediction
from dataset_utils import read_wikiqa_data, wiki_evaluation
from prompt_cot import cot_prompt
# Override bad SSL_CERT_FILE if set
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()
client = OpenAI()
### Module 1: Answer Generation VE
# Defining args parameter
# args = argparse.Namespace(
#     # style="standard", # used for few shot
#     style ="e-p", # used for consistency
#     annotation="std",
#     run_prediction=False,
#     run_length_test=False,
#     num_shot=5,
#     train_slice=0,
#     num_dev=5, # Test 10 instance, increase it later
#     dev_slice=0,
#     show_result=False,
#     model="gpt3.5",
#     show_prompt=False,
#     temperature =0.7,
#     engine = "gpt-3.5-turbo-0125",

# )

# # print(args)
# if args.style == "e-p":
#     args.helper = get_joint_prompt_helper(args.style)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, default="e-p")
    parser.add_argument('--annotation', type=str, default="std")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--dev_slice', type=int, default=0)
    parser.add_argument('--show_result', default=False, action='store_true')
    parser.add_argument('--model', type=str, default="gpt3.5")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--plot_consistency', default=False, action='store_true')
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument('--run_length_test', default=False)

    args, _ = parser.parse_known_args()
    args.helper = get_joint_prompt_helper(args.style)
    return args


def result_cache_name(args):
    return "misc/consistency_tr{}-{}_dv{}-{}_predictions_temp_{}.json".format(
        args.train_slice, args.train_slice + args.num_shot, args.dev_slice, args.num_dev,
        args.temperature)

### Module: COT Prediction
def in_context_manual_prediction(ex, prompt_template, engine, prompt_helper, length_test_only=False, n= 1):
    prompt, stop_signal = prompt_helper.prompt_for_joint_prediction(ex, prompt_template)

    temp = 0.7
    pred = {}
    responses = []
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=engine,
        n= n,
        temperature= temp
    )
    for choice in completion.choices:
        responses.append(choice.message.content)

    pred["responses"] = responses
    pred["id"] = ex["id"]
    pred["question"] = ex["question"]
    pred["right_answer"] = ex["answer"]

    return pred

def consistency(answers, rationales, predictions):
    answer_probs = {}
    # answer_prob_lists = {}
    choices = predictions['responses']
    for i, ans in enumerate(answers):
        # logprobs = np.array(choices[i]['logprobs']['token_logprobs'][prompt_tokens:])
        # prob = np.exp(np.mean(logprobs))
        if ans == 'NOT ENOUGH INFO':
            pass
        elif ans in answer_probs.keys():
            # answer_prob_lists[ans].append((i, prob))
            answer_probs[ans] += 1
        else:
            # answer_prob_lists[ans] = [(i, prob)]
            answer_probs[ans] = 1
    consistency = max(list(answer_probs.values()))/5
    final_aggregated_answer = sorted(answer_probs.items(), key=lambda item: item[1], reverse=True)[0][0]
    best_i = list(answer_probs.keys()).index(final_aggregated_answer)
    # prob_list = answer_prob_lists[final_aggregated_answer]
    # best_i = prob_list[np.argmax([a[1] for a in prob_list])][0]
    # Choosing rationale based on maximum length to get rationale with verifying questions 
    len_rationales = [len(i) for i in rationales]
    ind = len_rationales.index(max(len_rationales))
    final_aggregated_rationale = rationales[ind]
    return consistency, final_aggregated_answer, final_aggregated_rationale, best_i
    # return consistency, final_aggregated_answer

'''
def consistency(answers, rationales, predictions):
    answer_probs = {}
    # answer_prob_lists = {}
    choices = predictions['responses']
    for i, ans in enumerate(answers):
        # logprobs = np.array(choices[i]['logprobs']['token_logprobs'][prompt_tokens:])
        # prob = np.exp(np.mean(logprobs))
        if ans in answer_probs.keys():
            # answer_prob_lists[ans].append((i, prob))
            answer_probs[ans] += 1
        else:
            # answer_prob_lists[ans] = [(i, prob)]
            answer_probs[ans] = 1
    consistency = max(list(answer_probs.values()))/5
    final_aggregated_answer = sorted(answer_probs.items(), key=lambda item: item[1], reverse=True)[0][0]
    best_i = list(answer_probs.keys()).index(final_aggregated_answer)
    # prob_list = answer_prob_lists[final_aggregated_answer]
    # best_i = prob_list[np.argmax([a[1] for a in prob_list])][0]
    final_aggregated_rationale = rationales[best_i]
    return consistency, final_aggregated_answer, final_aggregated_rationale, best_i
    # return consistency, final_aggregated_answer
'''
# con, best_answer, best_rationale, best_i= consistency(answers, rationales, p)

def post_process_consistency(ex, p, args):
    answers, rationales = [], []
    for choice in p['responses']:
        # first split the rationales
        answer, rationale = args.helper.post_process_prediction(choice, change_rationale = False)
        answers.append(answer)
        rationales.append(rationale)
    # prompt_tokens = p['usage']['prompt_tokens']
    con, best_answer, best_rationale, best_i = consistency(answers, rationales, p)
    # new_p = p['choices'][best_i]
    new_p = {'response': p['responses'][best_i]}
    new_p['id'] = ex['id']
    new_p['question'] = ex['question']
    new_p['right_answer'] = ex['answer']
    new_p['consistency'] = con
    new_p['rationale'] = best_rationale
    new_p['answer'] = best_answer
    new_p['original_answers'] = answers
    new_p['original_rationales'] = rationales
    return con, new_p


def evaluate_manual_predictions(dev_set, predictions, style="p-e", do_print=False):
    acc_records = []
    rat_records = []
    f1_records, pre_records, rec_records = [], [], []
    # logprob_records = []
    # ansprob_records = []

    true_cons = []
    false_cons = []
    for idx, (ex, pred) in enumerate(zip(dev_set, predictions)):
        p_ans = pred['answer']
        p_rat = pred['rationale']
        acc, (f1, pre, rec), gt_ans = wiki_evaluation(p_ans, ex["answer"])
        acc_records.append(acc)
        rat_acc = False
        rat_records.append(rat_acc)
        f1_records.append(f1), pre_records.append(pre), rec_records.append(rec)
        # logprob_records.append(pred['joint_lobprob'])
        # ansprob_records.append(pred['answer_logprob'])
        if acc:
            true_cons.append(pred['consistency'])
        else:
            false_cons.append(pred['consistency'])

        if do_print:
            print("--------------{} EX {} RAT {} F1 {:.2f} CONS {:.2f}--------------".format(idx, acc, rat_acc, f1, pred['consistency']))
            print('question: ', ex['question'])
            for (i, answer) in enumerate(pred['original_answers']):
                rat = pred['original_rationales'][i]
                print(f'{i}: {rat} | {answer}')
            print('PR ANS:', p_ans)
            print('PR RAT:', p_rat)
            print('GT ANS:', gt_ans)
            print(json.dumps({'qas_id': ex['id'], 'answer': p_ans}))

    mean_of_array = lambda x: sum(x) / len(x)
    print("EX", mean_of_array(acc_records), "RAT", mean_of_array(rat_records))
    print("F1: {:.2f}".format(mean_of_array(f1_records)), 
            "PR: {:.2f}".format(mean_of_array(pre_records)),
            "RE: {:.2f}".format(mean_of_array(rec_records)))
    # print("Acc-Cov AUC: {:.2f}".format(f1auc_score(
    #         ansprob_records, acc_records)))
    
    cons = true_cons + false_cons
    print('consistencies: mean {} and std {}'.format(np.mean(cons), np.std(cons)))
    
    print('consistencies for true predictions: mean {} and std {}'.format(np.mean(true_cons), np.std(true_cons)))
    print('consistencies for false predictions: mean {} and std {}'.format(np.mean(false_cons), np.std(false_cons)))
    return true_cons, false_cons


### Final Function to use predictions for consistency

def test_few_shot_manual_prediction(args):
    print("Running prediction")
    train_set = read_wikiqa_data(f"data/train_subset.json", manual_annotation_style=args.style)
    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    print('len(train_set): ', len(train_set))
    dev_set = read_wikiqa_data(f"data/dev_sampled.json")
    dev_set = dev_set[args.dev_slice:(args.num_dev)]

    prompt, _ = args.helper.prompt_for_joint_prediction(dev_set[0], cot_prompt)
    print('prompt: ')
    print(prompt)
    if os.path.exists(result_cache_name(args)) and not args.run_length_test:
        predictions = read_json(result_cache_name(args))

    else:
        predictions = []    
        for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
            pred = in_context_manual_prediction(x, cot_prompt, engine=args.engine, prompt_helper=args.helper, length_test_only=args.run_length_test, n = args.num_shot)
            if pred != None:
                predictions.append(pred)
            else: #error, ending early
                args.num_dev = len(predictions) + args.dev_slice
                break
        
        # print("PREDICTIONS BEFORE SAVING", predictions)
        # print("*"*50)    
        dump_json(predictions, result_cache_name(args)) 
        # # save
      
    new_predictions, cons = [], []
    for i, p in enumerate(tqdm(predictions, total=len(predictions), desc="Verifying")):
        ex = dev_set[i]
        con, new_p = post_process_consistency(ex, p, args)
        cons.append(con)
        new_predictions.append(new_p)
    predictions = new_predictions 

    true_cons, false_cons = evaluate_manual_predictions(dev_set, predictions, args.style, do_print=True)

    cons = [p['consistency'] for p in predictions]
    plt.figure(figsize=(10,5))
    df = pd.DataFrame.from_dict({'label': ['correct']*len(true_cons) + ['incorrect']*len(false_cons) + ['overall']*(len(true_cons)+len(false_cons))\
        , 'consistency': true_cons + false_cons + cons})
    sns.displot(df, x="consistency", hue="label", kind="kde", fill=True)
    plt.savefig(f"log/consistency_2.png")
    

    return predictions, true_cons, false_cons