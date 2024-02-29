from triton_base_model import logger, TritonPythonModelBase
import numpy as np
import triton_python_backend_utils as pb_utils

from sentence_transformers import SentenceTransformer

class TritonPythonModel(TritonPythonModelBase):
    def model_init(self):
        model_id = self.model_base_path + '/m3e_base/'
        logger.info(f"model_id:{model_id}")
        self.model = SentenceTransformer(model_id).to('cuda')

    def handle_input(self, request):
        in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
        list_temp = in_0.as_numpy().astype(np.bytes_)
        #logger.info(f'in_0 server list_temp: {list_temp}, {type(list_temp)}')
        tmp_inputs = []
        for temp in list_temp:
            tmp_inputs.append(temp[0].decode('utf-8'))
        return "", tmp_inputs

    def model_predict(self, inputs):
        outputs = self.model.encode(inputs, normalize_embeddings=True).tolist()
        return outputs
    
    def handle_output(self, outputs):
        out_tensor = pb_utils.Tensor("OUTPUT0", np.asarray(outputs))
        return "", [out_tensor]
