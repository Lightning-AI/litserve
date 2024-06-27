import time
import litserve as ls


class SlowStreamAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.model = None
        self.items = ["One", "Two"]

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        while len(self.items) > 0:
            yield self.items.pop(0)
            time.sleep(2)

    def encode_response(self, output_stream):
        for output in output_stream:
            yield {"output": output}


if __name__ == "__main__":
    api = SlowStreamAPI()
    server = ls.LitServer(api, stream=True)
    server.run(generate_client_file=False)
