from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import tensorflow as tf
import numpy as np
from settings import HOST, PORT
from settings import TIMEOUT
from settings import INPUT, OUTPUT
from settings import EMPTY_TEXT
from settings import MODEL_NAME


class TfServing:
    def __init__(self):
        self.channel = implementations.insecure_channel(HOST, PORT)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            self.channel
        )
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = MODEL_NAME
        self.request_timeout = TIMEOUT

    def double_tensor_to_list(self, tensor):
        shape = [x.size for x in tensor.tensor_shape.dim]
        doubles = np.array(tensor.float_val)
        return doubles.reshape(shape)

    def call(self, data):
        tensor_proto = tf.contrib.util.make_tensor_proto(data, dtype=tf.float32)
        self.request.inputs[INPUT].CopyFrom(tensor_proto)
        predict_response = self.stub.Predict(self.request, timeout=self.request_timeout)
        output = self.double_tensor_to_list(predict_response.outputs[OUTPUT])
        # output of two time steps are often garbage
        output = np.squeeze(output[:, 2:, :])
        return output
