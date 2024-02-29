import collections
import re
import sys

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch
import os
import argparse
import flask
import json
import time
import logging
from threading import Thread
import multiprocessing as mp

torch.multiprocessing.set_start_method('spawn')

# from transformers_stream_generator import init_stream_support
# init_stream_support()

app = flask.Flask(__name__)
# python safe_check_server.py --model-id /home/wuzhiying/data/security/security_models/chinese_roberta_wwm_ext/ --device cuda --port 8300

parser = argparse.ArgumentParser(description='safty check server')
parser.add_argument('--model-id', default="", type=str)
# parase.add_argument('--model-type', default="LlamaForCausalLM")
parser.add_argument('--device', default="cpu", type=str)
parser.add_argument('--port', default=8001)
#parser.add_argument('--model_path',
#                    default="/baichuan/yiyu/safety_detector/outputs/multilabel_v04_relabeled/outputs/checkpoint-22000",
#                    type=str)

# lock = mp.Lock()
args = parser.parse_args()
print(args)


# if args.model_type =='chinese_roberta_wwm_ext':
#     #classify
#     from transformers import BertForSequenceClassification
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model =  BertForSequenceClassification.from_pretrained(model_id).to(args.device)
# else if args.model_type == "opus_mt_en_zh":
#     from transformers import AutoModelForSeq2SeqLM
#     #translate eng->chinese
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(args.device)
# else if args.model_type == "m3e_base":
#     #emb vec
#     from sentence_transformers import SentenceTransformer
#     model = SentenceTransformer(model_id).to(args.device)


def get_logger(name):
    loger = logging.getLogger('serverLog')
    loger.setLevel(logging.INFO)
    steram_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=f'logs/{name}.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s")
    file_handler.setFormatter(formatter)
    steram_handler.setFormatter(formatter)
    loger.addHandler(steram_handler)
    loger.addHandler(file_handler)
    return loger


logger = get_logger("safty_check_server")


def make_msg(code, msg, data={}):
    return {'code': code, 'msg': msg, 'data': data}


def get_k_v(data, key, default, type_):
    return default if key not in data else type_(data[key])


######
'''
    safe_check_request format:
        {
            user_id:int
            appid: int
            timestamp: int64
            size: int
            data_list:
                {
                    stat: int
                    model_name: string
                    model_version: string
                    type: string//chinese->eng\ eng->chinese
                    data: string
                    value: int //result
                }
        }
    translate_response format:
        {
            status:int
            message: string // info
            timestamp: int64 //  ms
            size: int // data_list len
            data_list:
                {
                     stat: int
                    model_name: string
                    model_version: string
                    type: string//chinese->eng\ eng->chinese
                    data: string
                    value: int //result
                }
        }
'''


class SafetyClassifier(object):
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.max_length = 512
        self._load_dicts()
        self._load_model()

        self.safe_conditions = [  # OR
            {'safe': 0.8, 'max_unsafe': 0.15},  # AND
            {'safe': 0.3, 'max_unsafe': 0.02},  # AND
        ]

        self.unsafe_conditions = [  # OR
            # Empty
        ]

    def detect(self, text_list, history, *args, **kwargs):
        if not text_list:
            return []
        history_prompts = [x['prompt'] for x in history if 'prompt' in x]
        if history and not history_prompts:
            print('history is not empty but no prompt in history', file=sys.stderr)
            print('history:', history, file=sys.stderr)
        history_scores = [self._predict_prompt(prompt) for prompt in history_prompts]
        result_list = []
        for prompt in text_list:
            safety_score, ordered_unsafe_predicts = self._predict_prompt(prompt)
            safety_score, ordered_unsafe_predicts = self._merge_scores([(safety_score, ordered_unsafe_predicts)] + history_scores)

            detect_info = {'safety_code': 0, 'ordered_unsafe_predicts': ordered_unsafe_predicts,
                           'safety_score': safety_score, 'sub_module': 'classifier'}
            if self._detect_safe(safety_score, ordered_unsafe_predicts):
                detect_info['safety_code'] = 1
            elif self._detect_unsafe(safety_score, ordered_unsafe_predicts):
                detect_info['safety_code'] = 2
            result_list.append(detect_info)
        return result_list

    def _detect_safe(self, safety_score, unsafe_predicts):
        match_conditions = []
        max_unsafe_score = unsafe_predicts[0]['score']
        for condition in self.safe_conditions:
            match_conditions.append(safety_score > condition['safe'] and
                                    max_unsafe_score < condition['max_unsafe'])
        return any(match_conditions)

    def _detect_unsafe(self, safety_score, unsafe_predicts):
        match_conditions = []
        max_unsafe_score = unsafe_predicts[0]['score']
        for condition in self.unsafe_conditions:
            match_conditions.append(safety_score < condition['safe'] and
                                    max_unsafe_score > condition['max_unsafe'])
        return any(match_conditions)

    def _split_text(self, text):
        """
        1. split text by punctuations
        2. merge sentences if len(sentence) < 20
        3. if a sentence contains quote, and length of words in quote > 5, split it
        """
        punctuations = ',!?.;，！？。；'
        quotations = '“”‘’<>《》[]【】()（）'
        quotations_dict = {quotations[i]: quotations[i + 1] for i in range(0, len(quotations), 2)}
        sentences = []
        start = 0
        for i, c in enumerate(text):
            if c in punctuations:
                sentences.append(text[start: i + 1])
                start = i + 1
        if start < len(text):
            sentences.append(text[start:])

        merged_sentences = []
        while sentences:
            s = sentences.pop(0)
            while sentences and len(s) + len(sentences[0]) < 20:
                s += sentences.pop(0)
            merged_sentences.append(s)
        if not merged_sentences:
            merged_sentences.append(text)

        # if a sentence contains quote, and length of words in quote > 5, split it
        def split_sentence_by_quotations(sentence):
            new_sentences = []
            for i, c in enumerate(sentence):
                if c in quotations_dict:
                    quote_end = sentence.find(quotations_dict[c], i + 1)
                    if quote_end > 0 and quote_end - i > 4:
                        new_sentences.append(sentence[:i])
                        new_sentences.append(sentence[i + 1: quote_end])
                        new_sentences.append(sentence[quote_end+1:])
                        return new_sentences
            return [sentence]

        new_merged_sentences = []
        for s in merged_sentences:
            new_merged_sentences.extend(split_sentence_by_quotations(s))

        # filter out sentences which do not contain any chinese characters or alphabets
        merged_sentences = []
        for s in new_merged_sentences:
            if re.search('[\u4e00-\u9fa5a-zA-Z]', s):
                merged_sentences.append(s)

        # print('Raw=====', text)
        # print('Merged=====', merged_sentences)
        return merged_sentences

    def _merge_scores(self, scores):
        """
        merge scores of same label
        """
        min_safe_score = 1.0
        merged_max_unsafe_scores = collections.defaultdict(lambda : 0.0)
        for safe_score, unsafe_scores in scores:
            min_safe_score = min(min_safe_score, safe_score)
            for unsafe_score in unsafe_scores:
                merged_max_unsafe_scores[unsafe_score['label']] = max(merged_max_unsafe_scores[unsafe_score['label']],
                                                                      unsafe_score['score'])
        unsafe_scores = [{'label': k, 'score': v} for k, v in merged_max_unsafe_scores.items()]
        sorted_unsafe_scores = sorted(unsafe_scores, key=lambda x: x['score'], reverse=True)
        return min_safe_score, sorted_unsafe_scores

    def _predict(self, text_list):
        result_list = self.classifier(text_list, batch_size=len(text_list))
        for result in result_list:
            safety_score = [res for res in result if res['label'] == 'LABEL_0'][0]['score']
            ordered_unsafe_predicts = [{'score': res['score'],
                                        'label': self.labels_list[int(res['label'].split('_')[1])]}
                                       for res in result if res['label'] != 'LABEL_0']
            yield safety_score, ordered_unsafe_predicts

    def _predict_prompt(self, prompt):
        segments = self._split_text(prompt)
        safety_score, ordered_unsafe_predicts = self._merge_scores(self._predict(segments))
        return safety_score, ordered_unsafe_predicts

    def _load_model(self):
        self.classifier = pipeline("text-classification", model=self.model,
                                   tokenizer=self.tokenizer, max_length=self.max_length,
                                   top_k=None, function_to_apply='sigmoid', device=0)

    def _load_dicts(self):
        model_args = json.loads(open(os.path.join(self.model_path, 'model_args.json'), 'r').read())
        self.labels_list = model_args['labels_list']
        return self.labels_list


# classify
classifier = SafetyClassifier(model_path=args.model_id, device=args.device)

#####
@app.route('/generate', methods=["GET", "POST"])
def generateInterface():
    pid = os.getpid()
    print(pid)
    req = json.loads(flask.request.get_data())
    print(req)
    user_id = get_k_v(req, 'user_id', 0, int)
    appid = get_k_v(req, 'appid', 0, int)
    timestamp = get_k_v(req, 'timestamp', 0, int)
    # size = get_k_v(req, 'size', 1, int)
    data_list = get_k_v(req, 'data_list', [], list)
    history = get_k_v(req, 'history', [], list)
    size = len(data_list)
    ######Get inputs
    input_strs = [data['data'].strip() for data in data_list]
    #######Generate Result
    result_list = classifier.detect(input_strs, history)
    #######Format to TranslateResponse
    response = dict()
    response['status'] = 0
    response['message'] = ''
    response['timestamp'] = int(time.time() * 1000)
    response['size'] = size
    response_data_list = []
    for i in range(size):
        response_dict = dict()
        response_dict['stat'] = 0
        response_dict['model_name'] = 'chinese_roberta_wwm_ext'
        response_dict['model_version'] = '0.0.1'
        # response_dict['type'] = "xx"
        response_dict['data'] = data_list[i]['data']
        response_dict['value'] = result_list[i]['safety_code']
        response_data_list.append(response_dict)
    response['data_list'] = response_data_list
    return json.dumps(response, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)
