# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import asyncio
import contextlib
from contextlib import asynccontextmanager
import inspect
from multiprocessing import Process, Manager, Queue, Pipe
from queue import Empty
import os
import shutil
from typing import Sequence
import uuid

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.security import APIKeyHeader

from litserve import LitAPI
from litserve.schemas.openai import UsageInfo, ChatCompletionRequest, ChatMessage, ChatCompletionResponseChoice, ChatCompletionResponse


# if defined, it will require clients to auth with X-API-Key in the header
LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY")


def single_loop(lit_api, device, worker_id, request_queue, request_buffer):
    while True:
        try:
            uid = request_queue.get(timeout=1.0)
            try:
                x_enc, pipe_s = request_buffer.pop(uid)
            except KeyError:
                continue
        except (Empty, ValueError):
            continue

        x = lit_api.decode_request(x_enc)
        y = lit_api.predict(x)
        y_enc = lit_api.encode_response(y)

        with contextlib.suppress(BrokenPipeError):
            pipe_s.send(y_enc)
 

def batched_loop(lit_api, device, worker_id, request_queue, request_buffer, max_batch_size):
    while True:
        inputs = []
        pipes = []
        for _ in range(max_batch_size):
            try:
                uid = request_queue.get(timeout=0.01)
                try:
                    x_enc, pipe_s = request_buffer.pop(uid)
                except KeyError:
                    continue
            except (Empty, ValueError):
                break

            x = lit_api.decode_request(x_enc)

            inputs.append(x)
            pipes.append(pipe_s)

        if not inputs:
            continue

        x = lit_api.batch(inputs)
        y = lit_api.predict(x)

        outputs = lit_api.unbatch(y)

        for pipe_s, y in zip(pipes, outputs):
            y_enc = lit_api.encode_response(y)

            with contextlib.suppress(BrokenPipeError):
                pipe_s.send(y_enc)
 

def inference_worker(lit_api, device, worker_id, request_queue, request_buffer, max_batch_size):
    lit_api.setup(device=device)

    if max_batch_size > 1:
        batched_loop(lit_api, device, worker_id, request_queue, request_buffer, max_batch_size)
    else:
        single_loop(lit_api, device, worker_id, request_queue, request_buffer)


def no_auth():
    pass


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if x_api_key != LIT_SERVER_API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API Key. Check that you are passing a correct 'X-API-Key' in your header."
        )


def setup_auth():
    if LIT_SERVER_API_KEY:
        return api_key_auth
    return no_auth


def cleanup(request_buffer, uid):
    with contextlib.suppress(KeyError):
        request_buffer.pop(uid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.request_queue = Queue()
    manager = Manager()
    app.request_buffer = manager.dict()

    # NOTE: device: str | List[str], the latter in the case a model needs more than one device to run
    for worker_id, device in enumerate(app.devices * app.workers_per_device):
        if len(device) == 1:
            device = device[0]
        process = Process(
            target=inference_worker,
            args=(app.lit_api, device, worker_id, app.request_queue, app.request_buffer, app.max_batch_size),
            daemon=False,
        )
        process.start()

    yield


async def get_from_pipe(pipe, timeout):
    if pipe.poll(timeout):
        return pipe.recv()
    return HTTPException(status_code=504, detail="Request timed out")


class LitServer:
    # TODO: add support for accelerator="auto", devices="auto"
    def __init__(self, lit_api: LitAPI, accelerator="cpu", devices=1, workers_per_device=1, timeout=30, max_batch_size=1):
        self.app = FastAPI(lifespan=lifespan)
        self.app.lit_api = lit_api
        self.app.workers_per_device = workers_per_device
        self.app.timeout = timeout
        self.app.max_batch_size = max_batch_size

        initial_pool_size = 100
        self.max_pool_size = 1000
        self.pipe_pool = [Pipe() for _ in range(initial_pool_size)]

        decode_request_signature = inspect.signature(lit_api.decode_request)
        encode_response_signature = inspect.signature(lit_api.encode_response)

        self.request_type = decode_request_signature.parameters["request"].annotation
        if self.request_type == decode_request_signature.empty:
            self.request_type = Request

        self.response_type = encode_response_signature.return_annotation
        if self.response_type == encode_response_signature.empty:
            self.response_type = Response

        if accelerator == "cpu":
            self.app.devices = [accelerator]
        elif accelerator in ["cuda", "gpu"]:
            device_list = devices
            if isinstance(devices, int):
                device_list = range(devices)
            self.app.devices = [self.device_identifiers(accelerator, device) for device in device_list]

        self.setup_server()

    def new_pipe(self):
        try:
            pipe_s, pipe_r = self.pipe_pool.pop()
        except IndexError:
            pipe_s, pipe_r = Pipe()
        return pipe_s, pipe_r

    def dispose_pipe(self, pipe_s, pipe_r):
        if len(self.pipe_pool) > self.max_pool_size:
            return
        self.pipe_pool.append(pipe_s, pipe_r)

    def device_identifiers(self, accelerator, device):
        if isinstance(device, Sequence):
            return [f"{accelerator}:{el}" for el in device]
        return [f"{accelerator}:{device}"]

    def setup_server(self):
        @self.app.get("/", dependencies=[Depends(setup_auth())])
        async def index(request: Request) -> Response:
            return Response(content="litserve running")

        # TODO: automatically apply when the API implements the predict method
        @self.app.post("/predict", dependencies=[Depends(setup_auth())])
        async def predict(request: self.request_type, background_tasks: BackgroundTasks) -> self.response_type:
            uid = uuid.uuid4()

            pipe_s, pipe_r = self.new_pipe()

            if self.request_type == Request:
                request_data = await request.json()
            else:
                request_data = request

            self.app.request_buffer[uid] = (request_data, pipe_s)
            self.app.request_queue.put(uid)

            background_tasks.add_task(cleanup, self.app.request_buffer, uid)

            data = await asyncio.get_running_loop().create_task(get_from_pipe(pipe_r, self.app.timeout))

            if type(data) == HTTPException:
                raise data

            return data

        # TODO: automatically apply when the API is a OpenAILitAPI, don't otherwise
        @self.app.post('/v1/chat/completions', dependencies=[Depends(setup_auth())])
        async def chat(request: ChatCompletionRequest, background_tasks: BackgroundTasks) -> ChatCompletionResponse:
            if request.stream:
                raise HTTPException(status_code=400, detail="Streaming not currently supported")

            if request.stop is not None:
                raise HTTPException(status_code=400, detail="Parameter stop not currently supported")

            if request.frequency_penalty:
                raise HTTPException(status_code=400, detail="Parameter frequency_penalty not currently supported")

            if request.presence_penalty:
                raise HTTPException(status_code=400, detail="Parameter presence_penalty not currently supported")

            if request.max_tokens is not None:
                raise HTTPException(status_code=400, detail="Parameter max_tokens not currently supported")

            if request.top_p != 1.0:
                raise HTTPException(status_code=400, detail="Parameter top_p not currently supported")

            uids = [uuid.uuid4() for _ in range(request.n)]
            pipes = []

            for uid in uids:
                pipe_s, pipe_r = self.new_pipe()

                self.app.request_buffer[uid] = (request, pipe_s)
                self.app.request_queue.put(uid)

                background_tasks.add_task(cleanup, self.app.request_buffer, uid)

                pipes.append(pipe_r)

            responses = []
            for pipe_r in pipes:
                data = await asyncio.get_running_loop().create_task(get_from_pipe(pipe_r, self.app.timeout))
                responses.append(data)

            choices = []

            usage = UsageInfo()
            for i, response in enumerate(responses):
                choices.append(
                    ChatCompletionResponseChoice(
                        index=i,
                        message=ChatMessage(role="assistant", content=response["text"]),
                        finish_reason=response.get("finish_reason", "stop"),
                    )
                )
                if "usage" in response:
                    task_usage = UsageInfo.parse_obj(response["usage"])
                else:
                    task_usage = UsageInfo()
                for usage_key, usage_value in task_usage.dict().items():
                    setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

            model = request.model or "litserve"
            return ChatCompletionResponse(model=model, choices=choices, usage=usage)

    def generate_client_file(self):
        src_path = os.path.join(os.path.dirname(__file__), "python_client.py")
        dest_path = os.path.join(os.getcwd(), "client.py")

        if os.path.exists(dest_path):
            return

        # Copy the file to the destination directory
        try:
            shutil.copy(src_path, dest_path)
            print(f"File '{src_path}' copied to '{dest_path}'")
        except Exception as e:
            print(f"Error copying file: {e}")

    def run(self, port=8000, log_level="info", **kwargs):
        self.generate_client_file()

        import uvicorn

        uvicorn.run(host="0.0.0.0", port=port, app=self.app, log_level=log_level, workers=1, **kwargs)
