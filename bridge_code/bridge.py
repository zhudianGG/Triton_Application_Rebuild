#new union Bridge Server
import torch
import os
import sys
import argparse
import traceback
import flask
import tqdm
import json
import numpy as np
import time
import logging
import requests
from flask import stream_with_context, request, Response, abort, jsonify
from threading import Thread 
import multiprocessing as mp

def get_logger(name=None):
    loger = logging.getLogger('serverLog')
    loger.setLevel(logging.INFO)
    steram_handler = logging.StreamHandler(sys.stdout)
    #file_handler=logging.FileHandler(filename=f'logs/{name}.log',mode='w',encoding='utf-8')
    #file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s")
    #file_handler.setFormatter(formatter)
    steram_handler.setFormatter(formatter)
    loger.addHandler(steram_handler)
    #loger.addHandler(file_handler)
    return loger
logger = get_logger("bridge")

from tritonclient.http import InferenceServerClient, InferInput  
from tritonclient.utils import *
app = flask.Flask(__name__)

parser = argparse.ArgumentParser(description='Server')
parser.add_argument('--model-type', required=True, 
                        choices=['m3e_base', 'opus_mt_en_zh', 'chinese_roberta_wwm_ext'], 
                        help='model type')
parser.add_argument('--triton_url', default="0.0.0.0:8000", 
                        help='model type')
parser.add_argument('--port', default=80, type=int)

args = parser.parse_args()

def get_k_v(data,key,default,type_):
    return default if key not in data else type_(data[key])

######
'''
    embedding_request format:
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
                    dim: int32
                    data: string 
                    value: [] float
                }
        }
    embedding_response format:
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
                    dim: int32
                    data: string 
                    value: [] float                  
                }
        }
'''
def embedding_inference(req, model_id):
    data_list = get_k_v(req,'data_list',[],list) 
    size = len(data_list)
    input_strs = []
    # input_strs.append("Hello")
    for i in range(size):
        input = data_list[i]['data'].strip()
        # print(f'input-data: {input}')
        input_strs.append(input)

    all_input = [bytes(in_, encoding = 'utf-8') for in_ in input_strs]
    in0 = np.array(all_input, dtype=np.bytes_).reshape([len(all_input), 1])
    inputs = []
    inputs.append(InferInput('INPUT0', in0.shape, np_to_triton_dtype(in0.dtype)))
    inputs[0].set_data_from_numpy(in0)
    outputs = []
    #outputs = (httpclient.InferRequestedOutput('OUTPUT0'))
    triton_client = InferenceServerClient(url=args.triton_url)
    outputs = triton_client.infer(model_name=model_id, inputs=inputs)
    result = outputs.as_numpy('OUTPUT0').astype(np.float32).tolist()

    response = dict()
    response['status'] = 0
    response['message'] = ''
    response['timestamp'] = int(time.time()*1000)
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
        response['dim'] = len(result[i])
        response_data_list.append(response_dict)
    response['data_list'] = response_data_list
    return response

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
    safe_check_response format:
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
def safety_check_inference(req, model_id):
    data_list = get_k_v(req,'data_list',[],list)
    input_strs = [data['data'].strip() for data in data_list]
    history = get_k_v(req, 'history', [], list)

    history_prompts = [x['prompt'] for x in history if 'prompt' in x]
    if history and not history_prompts:
        loger.warning('history is not empty but no prompt in history')
        loger.warning('history:', history)

    size = len(data_list)

    input_strs = [bytes(in_, encoding = 'utf-8') for in_ in input_strs]
    in0 = np.array(input_strs, dtype=np.bytes_).reshape([1, len(input_strs), 1])

    history_strs = [bytes(in_, encoding = 'utf-8') for in_ in history_prompts]
    '''
    history_strs_len = max(len(history_strs), 1)
    padding_str = bytes("", encoding = 'utf-8')
    padding_size = len(input_strs) * history_strs_len - len(history_strs)
    history_strs.extend([padding_str] * padding_size)
    '''
    in1 = np.array(history_strs, dtype=np.bytes_).reshape([1, len(history_strs), 1])

    inputs = []
    inputs.append(InferInput('input_strs', in0.shape, np_to_triton_dtype(in0.dtype)))
    inputs[0].set_data_from_numpy(in0)
    inputs.append(InferInput('history', in1.shape, np_to_triton_dtype(in1.dtype)))
    inputs[1].set_data_from_numpy(in1)

    triton_client = InferenceServerClient(url=args.triton_url)
    outputs = triton_client.infer(model_name=model_id, inputs=inputs)
    #logger.info(f"outputs:{outputs}")
    safety_code = outputs.as_numpy('safety_code').astype(np.int32).tolist()
    safe_score = outputs.as_numpy('safety_score').astype(np.float32).tolist()
    unsafe_category = outputs.as_numpy('unsafe_category').astype(np.object_).tolist()
    unsafe_score = outputs.as_numpy('unsafe_score').astype(np.float32).tolist()
    #logger.info(f"safety_code:{safety_code}, safe_score:{safe_score}, unsafe_category:{unsafe_category}, unsafe_score:{unsafe_score}")

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
        response_dict['data'] = data_list[i]['data']
        response_dict['value'] = safety_code[i]
        model_output = {'safe_score': safe_score[i], 'unsafe_category': unsafe_category[i].decode('utf-8'), 'unsafe_score': unsafe_score[i]}
        response_dict['model_output'] = model_output
        response_data_list.append(response_dict)
    response['data_list'] = response_data_list
    return response

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
def translation_inference(req, model_id):
    data_list = get_k_v(req, 'data_list', [], list)
    size = len(data_list)

    data_list_ = [data['data'] for data in data_list]
    data_type_list = [data['data_type'] for data in data_list]

    all_input = [bytes(in_, encoding = 'utf-8') for in_ in data_list_]
    in0 = np.array(all_input, dtype=np.bytes_).reshape([len(all_input), 1])
    inputs = []
    inputs.append(InferInput('data', in0.shape, np_to_triton_dtype(in0.dtype)))
    inputs[0].set_data_from_numpy(in0)
    all_input = [bytes(in_, encoding = 'utf-8') for in_ in data_type_list]
    in1 = np.array(all_input, dtype=np.bytes_).reshape([len(all_input), 1])
    inputs.append(InferInput('data_type', in1.shape, np_to_triton_dtype(in1.dtype)))
    inputs[1].set_data_from_numpy(in1)

    outputs = []
    #outputs = (httpclient.InferRequestedOutput('OUTPUT0'))
    triton_client = InferenceServerClient(url=args.triton_url)
    outputs = triton_client.infer(model_name=model_id, inputs=inputs)
    result = outputs.as_numpy('OUTPUT0').astype(np.object_).tolist()

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
        response_dict['value'] = result[i].decode('utf-8')
        response_data_list.append(response_dict)
    response['data_list'] = response_data_list
    return response


@app.errorhandler(500)
def resource_not_found(e):
    return jsonify(error=str(e)), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        start_time = time.time()
        model_type = args.model_type

        req = json.loads(flask.request.get_data())
        user_id = get_k_v(req,'user_id',0,int)
        appid = get_k_v(req, 'appid', 0, int)
        timestamp = get_k_v(req, 'timestamp', 0, int)
        # size = get_k_v(req, 'size', 1, int)

        if model_type == 'm3e_base':
            results = embedding_inference(req, model_type)
        elif model_type == "opus_mt_en_zh":
            results = translation_inference(req, model_type)
        elif model_type == 'chinese_roberta_wwm_ext':
            results = safety_check_inference(req, model_type)  

        end_time = time.time()
        logger.info(f"generate {model_type}  cost:{int((end_time-start_time) * 1000)}ms")
        return json.dumps(results, ensure_ascii=False)
    except Exception as ex:
        logger.warning(f"generate error: {ex}  \n {traceback.format_exc()}")
        abort(500, description=f"{ex}")

@app.route('/health', methods=['GET'])
def health():
    triton_client = InferenceServerClient(url=args.triton_url)
    try:
        if not triton_client.is_server_live() or  \
            not triton_client.is_server_ready() or \
            not triton_client.is_model_ready(args.model_type):
            abort(500, description=f" triton error")
    except Exception as ex:
        logger.info(f"triton error")
        abort(500, description=f" triton error")
    return ""

if __name__ == '__main__':
    from gevent import monkey
    monkey.patch_all()
    triton_client = InferenceServerClient(url=args.triton_url)
    while True:
        try:
            if triton_client.is_server_ready() and triton_client.is_model_ready(args.model_type):
                break
        except Exception as ex:
            logger.info(f"wait for triton start")
            time.sleep(1)
            continue
    app.run(host='0.0.0.0',port=args.port)