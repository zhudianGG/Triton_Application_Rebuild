import numpy as np
import json
import collections
import re
import sys
import os
from typing import List, Tuple
import urllib

import triton_python_backend_utils as pb_utils

from triton_base_model import logger, TritonPythonModelBase
from transformers import AutoTokenizer, pipeline


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

    def process(self, data_list, data_type_list):
        """
        :param data_list: list of dict, each dict has key 'data' and 'data_type'
        """
        # preprocess
        '''
        for data in data_list:
            data['data'] = self._preprocess(data['data'])
        # choose data to be translated
        batch_data_list = [self._split(data['data']) for data in data_list]
        data_type_list = [data['data_type'] for data in data_list]
        '''

        data_list = [self._preprocess(data) for data in data_list]
        # choose data to be translated
        batch_data_list = [self._split(data) for data in data_list]

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


class TritonPythonModel(TritonPythonModelBase):
    def model_init(self):
        model_id = self.model_base_path + '/opus_mt_en_zh/'
        logger.info(f"model_id:{model_id}")
        self.translator = Translator(model_name_or_path=model_id, device="cuda")

    def handle_input(self, request):
        in_0 = pb_utils.get_input_tensor_by_name(request, "data")
        list_temp = in_0.as_numpy().astype(np.bytes_)
        # logger.info(f'in_0 server list_temp: {list_temp}, {type(list_temp)}')
        data = []
        for temp in list_temp:
            data.append(temp[0].decode('utf-8'))


        in_1 = pb_utils.get_input_tensor_by_name(request, "data_type")
        list_temp = in_1.as_numpy().astype(np.bytes_)
        # logger.info(f'in_0 server list_temp: {list_temp}, {type(list_temp)}')
        data_type = []
        for temp in list_temp:
            data_type.append(temp[0].decode('utf-8'))

        return "", [(data, data_type)]

    def model_predict(self, inputs):
        #logger.info(f"inputs: {inputs}")
        data_list = []
        data_type_list = []
        output_idx = [0]
        for input_ in inputs:
            data_list.extend(input_[0])
            data_type_list.extend(input_[1])
            output_idx.append(len(data_list))
        result = self.translator.process(data_list, data_type_list)
        ret_list = []
        for i in range(len(output_idx) -1):
            ret_list.append(result[output_idx[i]:output_idx[i+1]])
        #logger.info(f"data_list: {data_list} data_type_list:{data_type_list} result:{result} ret_list:{ret_list}")
        return ret_list
    
    def handle_output(self, outputs):
        #logger.info(f"outputs: {outputs}")
        assert len(outputs) == 1, f"{outputs}"
        out_tensor = pb_utils.Tensor("OUTPUT0", np.asarray(outputs[0], dtype=object))
        return "", [out_tensor]

