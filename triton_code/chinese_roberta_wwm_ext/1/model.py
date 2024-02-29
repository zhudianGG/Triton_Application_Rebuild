import numpy as np
import json
import collections
import re
import sys
import os

import triton_python_backend_utils as pb_utils

from triton_base_model import logger, TritonPythonModelBase
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

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

        self.skip_detection_pattern = re.compile(r'[ux%#\\&]')

    def detect(self, text_list, history_prompts, *args, **kwargs):
        if not text_list:
            return []
        '''
        history_prompts = [x['prompt'] for x in history if 'prompt' in x]
        if history and not history_prompts:
            print('history is not empty but no prompt in history', file=sys.stderr)
            print('history:', history, file=sys.stderr)
        '''
        history_scores = [self._predict_prompt(prompt) for prompt in history_prompts]
        result_list = []
        for prompt in text_list:
            safety_score, ordered_unsafe_predicts = self._predict_prompt(prompt)
            safety_score, ordered_unsafe_predicts = self._merge_scores([(safety_score, ordered_unsafe_predicts)] + history_scores)
            detect_info = {'safety_code': 0, 'ordered_unsafe_predicts': ordered_unsafe_predicts,
                           'safety_score': safety_score, 'sub_module': 'classifier'}
            should_skip_detection = self.skip_detection_pattern.search(prompt) is not None
            # print(f'should skip detection: {should_skip_detection}')
            if should_skip_detection:
                pass
            elif self._detect_safe(safety_score, ordered_unsafe_predicts):
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
        if not merged_sentences:
            merged_sentences.append('')
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



class TritonPythonModel(TritonPythonModelBase):
    def model_init(self):
        model_id = self.model_base_path + '/chinese_roberta_wwm_ext/'
        logger.info(f"model_id:{model_id}")
        self.classifier = SafetyClassifier(model_path=model_id, device="cuda")

    def handle_input(self, request):
        in_0 = pb_utils.get_input_tensor_by_name(request, "input_strs")
        list_temp = in_0.as_numpy().astype(np.bytes_)
        # logger.info(f'in_0 server list_temp: {list_temp}, {type(list_temp)}')
        tmp_inputs = []
        for temp_ in list_temp:
            for temp in temp_:
                tmp_inputs.append(temp[0].decode('utf-8'))

        in_1 = pb_utils.get_input_tensor_by_name(request, "history")
        list_temp = in_1.as_numpy().astype(np.bytes_)
        #logger.info(f'in_1 server list_temp: {list_temp}, {type(list_temp)}')
        history = []
        for temp_ in list_temp:
            #logger.info(f'in_1 server temp: {temp}, {type(temp)}')
            for temp in temp_:
                h = temp[0].decode('utf-8')
                if h != "":
                    history.append(h)

        return "", [(tmp_inputs, history)]

    def model_predict(self, inputs):
        result_list = []
        for input_ in inputs:
            result = self.classifier.detect(input_[0], input_[1])
            result_list.append(result)
        #logger.info(f"inputs:{inputs}, result_list:{result_list}")
        return result_list
    
    def handle_output(self, outputs):
        safety_code = []
        safety_score = []
        unsafe_category = []
        unsafe_score = []
        #logger.info(f"outputs:{outputs}")
        for results in outputs:
            for result in results:
                safety_code.append(result['safety_code'])
                safety_score.append(result['safety_score'])
                unsafe_predict = result['ordered_unsafe_predicts'][0]
                unsafe_category.append(unsafe_predict['label'])
                unsafe_score.append(unsafe_predict['score'])
        safety_code = pb_utils.Tensor("safety_code", np.asarray(safety_code))
        safety_score = pb_utils.Tensor("safety_score", np.asarray(safety_score))
        unsafe_category = pb_utils.Tensor("unsafe_category", np.asarray(unsafe_category, dtype=object))
        unsafe_score = pb_utils.Tensor("unsafe_score", np.asarray(unsafe_score))
        return "", [safety_code, safety_score, unsafe_category, unsafe_score]
