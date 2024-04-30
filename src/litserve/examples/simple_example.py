import litserve as ls
import numpy as np


class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # Set up the model, so it can be called in `predict`.
        self.model = lambda x: x**2

    def decode_request(self, request):
        # Convert the request payload to your model input.
        return request["input"]

    def predict(self, x):
        # Run the model on the input and return the output.
        return self.model(x)

    def encode_response(self, output):
        # Convert the model output to a response payload.
        return {"output": output}


class SimpleBatchedAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = lambda x: x**2

    def decode_request(self, request):
        return np.asarray(request["input"])

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}