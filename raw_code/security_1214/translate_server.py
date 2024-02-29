import argparse
import json
import logging
import re
import time
import torch
import os
from typing import List, Tuple

import flask
from transformers import AutoTokenizer, pipeline

import urllib

# from transformers_stream_generator import init_stream_support
# init_stream_support()

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("BC_GPU_MEMORY_FRACTION", '1.0')))
app = flask.Flask(__name__)
# python translate_server.py --model-id /home/wuzhiying/data/security/security_models/opus_mt_en_zh/ --device cuda --port 8300

parser = argparse.ArgumentParser(description='translation server')
parser.add_argument('--model-id', default="", type=str)
# parase.add_argument('--model-type', default="LlamaForCausalLM")
parser.add_argument('--device', default="cpu", type=str)
parser.add_argument('--port', default=8001)
# lock = mp.Lock()
args = parser.parse_args()
print(args)
model_id = args.model_id

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


logger = get_logger("translation_server")


def make_msg(code, msg, data={}):
    return {'code': code, 'msg': msg, 'data': data}


def get_k_v(data, key, default, type_):
    return default if key not in data else type_(data[key])


######
'''
    translate_request format:
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
                    value: string //result
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
                    value: string //result
                }
        }
'''


class Translator:
    def __init__(self, model_name_or_path, device='cpu', **kwargs):
        self.re_alphanum = re.compile(r'[a-zA-Z0-9]')
        self.re_alpha = re.compile(r'[a-zA-Z]')
        self.re_punc = re.compile(r'[ ,.?!\-\'\"]')
        self.re_chinese = re.compile(r'[\u4e00-\u9fff]')
        self.max_length = 512

        self.prompt_condition = lambda words_cnt, alpha_cnt: words_cnt >= 2 and alpha_cnt >= 7
        self.answer_condition = lambda words_cnt, alpha_cnt: words_cnt >= 5

        self.translator_en_to_zh = pipeline("translation_en_to_zh",
                                            model=model_name_or_path, device=0, max_length=self.max_length)

        self.TYPE_PROMPT = 'prompt'
        self.TYPE_ANSWER = 'answer'

        self.condition_dict = {
            self.TYPE_PROMPT: self.prompt_condition,
            self.TYPE_ANSWER: self.answer_condition,
            'default': lambda words_cnt, alpha_cnt: False  # always False
        }

    def process(self, data_list):
        """
        :param data_list: list of dict, each dict has key 'data' and 'data_type'
        """
        # preprocess
        for data in data_list:
            data['data'] = self._preprocess(data['data'])
        # choose data to be translated
        batch_data_list = [self._split(data['data']) for data in data_list]
        data_type_list = [data['data_type'] for data in data_list]
        need_tobe_translated = []
        for i, (segments, data_type) in enumerate(zip(batch_data_list, data_type_list)):
            match_condition = self.condition_dict.get(data_type, self.condition_dict['default'])
            for j, (segment, t) in enumerate(segments):
                if t == 'en':
                    words_cnt = len([1 for w in segment.split(' ') if any([self.re_alpha.match(c) for c in w])])
                    alpha_cnt = len([1 for c in segment if self.re_alpha.match(c)])
                    if match_condition(words_cnt, alpha_cnt):
                        need_tobe_translated.append({'text': segment, 'index': (i, j)})
        # translate
        translated = self._translate(need_tobe_translated)
        # merge
        result_list = [[segment for segment, t in segments] for segments in batch_data_list]
        for data in translated:
            i, j = data['index']
            result_list[i][j] = data['text']
        result_list = [''.join(segments) for segments in result_list]
        return result_list
    
    def _preprocess(self, text):
        """
        convert unicode / urlencode(%) / utf16 to unicode
        """
        pattern_replace_map = {
            r'\\u[0-9a-fA-F]{4}': lambda match: chr(int(match[2:], 16)),
            r'(?:%[0-9a-fA-F]{2})+': lambda match: urllib.parse.unquote(match),
            r'\&#x[0-9A-Fa-f]+;': lambda match: chr(int(match[3:-1], 16)),
            r'(?:\\x[0-9a-fA-F]{2})+': lambda match: bytes.fromhex(match.replace(r'\x',  '')).decode('utf8'),
            r'(?:\\[0-9a-fA-F]{3})+': lambda match: bytes([int(x, 8) for x in match.split('\\')[1:]]).decode('utf8')
        }
        for pattern, replace_func in pattern_replace_map.items():
            for match in re.findall(pattern, text):
                org_text = text
                try:
                    text = text.replace(match, replace_func(match))
                except Exception as e:
                    text = org_text
        return text

    def _translate(self, need_tobe_translated):
        """
        translate English words into Chinese words.
        @param need_tobe_translated: list of dict, each dict has key 'text' and 'index'
        """
        if len(need_tobe_translated) == 0:
            return need_tobe_translated
        text_list = [data['text'][:self.max_length] for data in need_tobe_translated]
        result_list = self.translator_en_to_zh(text_list, batch_size=len(text_list))
        for i, result in enumerate(result_list):
            translated_text = result['translation_text']
            # It succeeds only if it contains Chinese.
            if self.re_chinese.search(translated_text):
                need_tobe_translated[i]['text'] = translated_text
        return need_tobe_translated

    def _split(self, text) -> List[Tuple]:
        """
        split prompt into Chinese and English segments.
        """
        segments = []
        buffer = ""
        for c in text:
            # if c is English
            if self.re_alphanum.match(c) or self.re_punc.match(c):
                buffer += c
            # else
            else:
                if buffer != "":
                    segments.append((buffer, 'en'))
                    buffer = ""
                segments.append((c, 'zh'))
        if buffer != "":
            segments.append((buffer, 'en'))
        return segments


translator = Translator(model_name_or_path=args.model_id, device=args.device)

#####
@app.route('/generate', methods=["GET", "POST"])
def generateInterface():
    req = json.loads(flask.request.get_data())
    print(req)
    user_id = get_k_v(req, 'user_id', 0, int)
    appid = get_k_v(req, 'appid', 0, int)
    timestamp = get_k_v(req, 'timestamp', 0, int)
    # size = get_k_v(req, 'size', 1, int)
    data_list = get_k_v(req, 'data_list', [], list)
    size = len(data_list)

    #######Generate Result
    result = translator.process(data_list)
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
        response_dict['model_name'] = 'opus_mt_en_zh'
        response_dict['model_version'] = '0.0.1'
        response_dict['type'] = "Eng->Chinese"
        response_dict['data'] = data_list[i]['data']
        response_dict['value'] = result[i]
        response_data_list.append(response_dict)
    response['data_list'] = response_data_list
    return json.dumps(response, ensure_ascii=False)


app.run(host='0.0.0.0', port=args.port)
