# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import json
import triton_python_backend_utils as pb_utils
import torch
import logging
import sys
import json
import os
import time
import traceback

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
logger = get_logger("infer")

memory_fraction=float(os.environ.get("BC_GPU_MEMORY_FRACTION", '1.0'))
logger.info(f"memory_fraction: {memory_fraction}")
torch.cuda.set_per_process_memory_fraction(memory_fraction)


class TritonPythonModelBase:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        logger.info(f"model_config: {self.model_config}")
        self.max_batch_size = max(self.model_config['max_batch_size'], 1)

        '''
        self.input0_config = pb_utils.get_output_config_by_name(
            self.model_config, "INPUT0")
        self.input0_dtype = pb_utils.triton_string_to_numpy(self.input0_config['data_type'])

        self.output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "OUTPUT0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(self.output0_config['data_type'])
        '''
        #self.model_base_path = args['model_repository'] + '/model/'
        self.model_base_path = os.environ.get("MODEL_BASE_PATH")
        self.model_init()

    def model_init(self):
        raise NotImplementedError()

    def handle_input(self, request):
        raise NotImplementedError()

    def model_predict(self, inputs):
        raise NotImplementedError()

    def handle_output(self, outputs): 
        raise NotImplementedError()

    @classmethod
    def get_split(cls, l, n):
        return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]
    
    def execute(self, requests):
        """ This function is called on inference request.
        """
        responses = [None] * len(requests)
        all_input = []
        request_split = []
        for idx, request in enumerate(requests):
            pre_all_input_len = len(all_input)
            # logger.info(f"request:{request}")
            if request.is_cancelled():#not support before 23.10
                responses[idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Message", pb_utils.TritonError.CANCELLED))
                inputs = []
            else:
                try:
                    status, inputs = self.handle_input(request)
                except Exception as ex:
                    inputs = []
                    status = f"some err: {ex}"
                    logger.warning(f"input handle error: {status} \n {traceback.format_exc()}")
                if status != "":
                    responses[idx] = pb_utils.InferenceResponse(
                                        error=pb_utils.TritonError(f"input handle error: {status}"))
            all_input.extend(inputs)
            request_split.append((pre_all_input_len, len(all_input)))
        all_output = []
        
        for par_input in self.get_split(all_input, self.max_batch_size):
            try:
                outputs = self.model_predict(par_input)
            except Exception as ex:
                logger.warning(f"model_predict error: {ex} \n {traceback.format_exc()}")
                for idx in range(len(responses)):
                    responses[idx] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"model_predict error: {ex}"))
                return responses
            #logger.info(f'outputs:{outputs}')
            all_output.extend(outputs)
        
        for idx, (start, end) in enumerate(request_split):
            if end - start > 0:
                try:
                    part_outputs = all_output[start:end]
                    #logger.info(f'part_outputs:{part_outputs}')
                    status, out_tensor = self.handle_output(part_outputs)
                    if status == "":
                        responses[idx] = pb_utils.InferenceResponse(out_tensor)
                except Exception as ex:
                    status = f"some err: {ex}"
                    logger.warning(f"output handle error: {status} \n {traceback.format_exc()}")
                
                if status != "":
                    responses[idx] = pb_utils.InferenceResponse(
                                        error=pb_utils.TritonError(f"output handle error: {status}"))
        return responses

