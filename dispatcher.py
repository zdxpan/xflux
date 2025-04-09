import argparse
import json
import os
import torch

# from exception                      import BadRequest, MemoryExhaustedException, ParallelismLimitedException
from log                            import logger, statistical_runtime
from flux_pipe                      import FluxPipeline
from tensor_util                    import RESIZER
from queue                          import Queue
from threading                      import Condition, Thread, Lock, Semaphore
from input_checker                  import FluxParameterCheckNormalizer, convert_args_to_dict
from tile_flux_v2                   import LazyFluxPipeline

class BadRequest(Exception):
    pass


class MemoryExhaustedException(Exception):
    pass


class ParallelismLimitedException(Exception):
    pass


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)


class Dispatcher():
    def __call__(self, task, args):
        if not self.semaphore.acquire(blocking=False):
            raise ParallelismLimitedException('Reached diffusion parallelism limit')
        try:
            # Enter the processor one by one
            self.mutex.acquire()
            try:
                self.queue_in.put({
                    'task': task,
                    'args': args
                })
                result = self.queue_out.get()
                if isinstance(result, Exception):
                    raise result
            finally:
                self.mutex.release()

            return result
        except Exception as e:
            if ('CUDA out of memory.' in str(e)):
                e = MemoryExhaustedException('CUDA out of memory')
            raise e
        finally:
            self.semaphore.release()

    def __init__(self, opt):
        self.semaphore = Semaphore(opt.parallelism)
        self.processor = None
        self.queue_in = Queue()
        self.queue_out = Queue()
        self.mutex = Lock()
        self.ready = Condition()


        with self.ready:
            Thread(target=self.loop, args=(opt,)).start()
            self.ready.wait()
            logger.debug('Processor thread ready to work')

    def loop(self, opt):
        self.processor = Processor(opt)
        with self.ready:
            self.ready.notify()
        while True:
            result = None
            item = self.queue_in.get()
            if item is None:
                # None is used as the exit signal
                break
            task = item['task']
            args = item['args']  # multidict~
            try:
                # opt = self.load_args(args)
                statistical_runtime.reset_collection()
                args = convert_args_to_dict(args)
                param_normer = FluxParameterCheckNormalizer()
                opt = param_normer.normalize(args)
                logger.set_trace_id(opt.trace_id)
                logger.info(f'{task} with options:', opt)
                if task == 'generate':
                    result = self.processor(opt)
                elif task == 'train':
                    result = self.processor.train(opt)
                else:
                    raise Exception(f'Invalid task {task}')
            except Exception as e:
                result = e
            finally:
                self.queue_out.put(result)

    def stop(self):
        self.queue_in.put(None)


class Processor():
    def __init__(self, opt):
        statistical_runtime.reset_collection()
        self.device = get_device()
        self.dtype = torch.bfloat16 if opt.precision == 'half' else torch.float32
        # self.pipeline = FluxPipeline(
        #     opt.model,
        #     self.device,
        #     self.dtype
        # )
        self.pipeline = LazyFluxPipeline().load()
        # or 
        # text_encoder = PromptEncoder(
        #     base=DEFAULT_INDEX["flux"],
        #     device=device,
        #     dtype=torch.bfloat16,
        # )
        # f1pipe = FluxPipeline(base = DEFAULT_INDEX['flux']['root'], device=device, dtype=torch.bfloat16,esrgan=None, 
        #             controlnet=None, encoder_hid_proj=None, 
        #             prompt_encoder=text_encoder, annotator=None
        #         )
        # f1pipe.initiate()


    def __call__(self, opt):
        return self.pipeline(opt)
        
    def train(self, opt):
        return self.pipeline.train(opt)
