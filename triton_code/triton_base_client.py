from functools import partial
import argparse
import numpy as np
import time
import sys
# import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import re
from tritonclient.utils import *
import json
import requests

def http_req():
    input_str = '我是中国人'
    input_str = '你好'

    request_data = {
        "id": "32",
        "inputs": [
            {
                "name": "INPUT0",
                "datatype": "BYTES",
                "shape": [1,1],
                "data": [input_str]
            }
        ],
        "outputs": [{"name": "OUTPUT0"}, {"name": "OUTPUT0"}]
    }
    res = requests.post(url="http://localhost:8000/v2/models/m3e_base/versions/1/infer",json=request_data).json()
    print(res)

    res = requests.post(url="http://localhost:8000/v2/models/opus_mt_en_zh/versions/1/infer",json=request_data).json()
    print(res)

    res = requests.post(url="http://localhost:8000/v2/models/chinese_roberta_wwm_ext/versions/1/infer",json=request_data).json()
    print(res)
    #{'id': '42', 'model_name': 'm3e_base', 'model_version': '1', 'outputs': [{'name': 'OUTPUT0', 'datatype': 'FP64', 'shape': [1, 768], 'data': [0.05524080619215965,...]}]} 

if __name__ == '__main__':
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8006.')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=True,
                        help='model_name')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help='mode input')

    FLAGS = parser.parse_args()
    model_name = FLAGS.model_name #'chinese_roberta_wwm_ext'

    try:
        # triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
        #                                                  verbose=FLAGS.verbose)
        print(f'url: {FLAGS.url}')
        triton_client = httpclient.InferenceServerClient(url=FLAGS.url,
                                                             verbose=False)
        model_config = triton_client.get_model_config(model_name)
        print(f'model_config:{model_config}')

    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    
    if not triton_client.is_model_ready(model_name):
        print("FAILED : is_model_ready")
        sys.exit(1)
    # Infer
    inputs = []
    outputs = []
    all_input = FLAGS.input.split(";")
    all_input = [bytes(in_, encoding = 'utf-8') for in_ in all_input]
    ina = np.array(all_input, dtype=np.bytes_)
    #ina = np.array(bytes(stra, encoding = 'utf-8'), dtype=np.bytes_)
    print(f"ina :{ina}")
    #inputs.append(httpclient.InferInput('INPUT0', [1, len(ina)], np_to_triton_dtype(ina.dtype)))
    inputs.append(httpclient.InferInput('INPUT0', [len(ina)], np_to_triton_dtype(ina.dtype)))
    #ina = ina.reshape([1,len(ina)])
    ina = ina.reshape([len(ina)])
    #print(f"ina :{ina}")
    inputs[0].set_data_from_numpy(ina)
    #outputs.append(httpclient.InferRequestedOutput('OUTPUT0'))
    # Inference call
    # user_data = []
    # triton_client.async_infer(model_name=model_name,
    #                         inputs=inputs,
    #                         callback=partial(callback, user_data),
    #                         outputs=outputs,
    #                         client_timeout=FLAGS.client_timeout)
    outputs = triton_client.infer(model_name=model_name,
                            inputs=inputs)
    # print(outputs.as_numpy('OUTPUT0').astype(np.object_))
    outputs2 = outputs.as_numpy('OUTPUT0').astype(np.object_)
    for output in outputs2:
        print(output.decode('utf-8'))
    # print(f"====user_data:{outputs}")

#python3 test_client.py -m opus_mt_en_zh -i '{"data": "Hello World!"};{"data": "Hello World!"}'
