from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,PhrasalConstraint,TextIteratorStreamer
import torch
import os
import argparse
import flask
import tqdm
import json
import numpy as np 
import time
import logging
from flask import stream_with_context, request, Response
from threading import Thread
import multiprocessing as mp
# from transformers_stream_generator import init_stream_support
# init_stream_support()

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("BC_GPU_MEMORY_FRACTION", '1.0')))
app = flask.Flask(__name__)
# python embedding_server.py --model-id /home/wuzhiying/data/security/security_models/m3e_base/ --device cuda --port 8300

parser = argparse.ArgumentParser(description='embedding server')
parser.add_argument('--model-id',default="", type=str)
# parase.add_argument('--model-type', default="LlamaForCausalLM")
parser.add_argument('--device',default = "cpu", type=str)   
parser.add_argument('--port',default = 8001)
# lock = mp.Lock()
args = parser.parse_args()
print(args)
model_id = args.model_id 
from sentence_transformers import SentenceTransformer#translate eng->chinese
model = SentenceTransformer(model_id).to(args.device)

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
    file_handler=logging.FileHandler(filename=f'logs/{name}.log',mode='w',encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s")
    file_handler.setFormatter(formatter)
    steram_handler.setFormatter(formatter)
    loger.addHandler(steram_handler)
    loger.addHandler(file_handler)
    return loger
logger = get_logger("embedding_server")


def make_msg(code,msg,data={}):
    return {'code':code,'msg':msg,'data':data}
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
#####
@app.route('/generate',methods=["GET","POST"])
def generateInterface():
    req = json.loads(flask.request.get_data())
    print(req)
    user_id = get_k_v(req,'user_id',0,int)
    appid = get_k_v(req, 'appid', 0, int)
    timestamp = get_k_v(req, 'timestamp', 0, int)
    # size = get_k_v(req, 'size', 1, int)
    data_list = get_k_v(req,'data_list',[],list) 
    size = len(data_list)
    ######Get inputs
    input_strs = []
    # input_strs.append("Hello")
    for i in range(size):
        input = data_list[i]['data'].strip()
        # print(f'input-data: {input}')
        input_strs.append(input)

    #######Generate Result
    result = model.encode(input_strs, normalize_embeddings=True).tolist()
    # result = np.zeros([2, 768], dtype=np.float32).tolist()
    # result = result.numpy()
    #######Format to TranslateResponse 
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
    # print(response)
    return json.dumps(response)

app.run(host='0.0.0.0',port=args.port)

