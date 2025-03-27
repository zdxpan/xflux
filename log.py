import datetime
import functools
import json
import threading
import time
import traceback
import uuid

from opentelemetry import trace, metrics
from opentelemetry.trace import Span

LEVEL = {
    'error': 0,
    'warn': 1,
    'info': 2,
    'debug': 3
}

SYMBOL = {
    'error': '!!',
    'warn': '!!',
    'info': '>>',
    'debug': '##'
}

TRACER = trace.get_tracer("jssd.tracer")
METER = metrics.get_meter('jssd.meter')
METER_COUNTER = METER.create_counter('jssd.meter.counter')


class Logger():
    context = threading.local()

    def __call__(self, *message, level='info', stack=''):
        if LEVEL[level] > self.level:
            return

        timestamp = datetime.datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S")
        message = ' '.join(map(
            lambda x: str(x),
            message
        ))
        trace_id = self.context.trace_id if hasattr(self.context, 'trace_id') else ''
        req_id = self.context.req_id if hasattr(self.context, 'req_id') else ''
        if self.format == 'json':
            print(json.dumps({
                'timestamp': timestamp,
                'level': level.upper(),
                # 'thread': '',
                # 'class': '',
                'message': message,
                'stack_trace': stack,
                # 'handleTime': '',
                # 'timeUnit': '',
                'trace_id': trace_id,
                'req_id': req_id,
            }), flush=True)
        else:
            print(f'[{timestamp}] {SYMBOL[level]} {message}' + (f' @{id}' if id else ''), flush=True)
            if stack != '':
                print(stack, flush=True)

    def __init__(self):
        self.format = 'json'
        self.level = LEVEL['debug']
        self.timezone = None

    def config(self, format='json', level='debug', timezone=None):
        self.format = format
        self.level = LEVEL[level]
        if timezone != None:
            self.timezone = datetime.timezone(datetime.timedelta(hours=timezone))
        else:
            self.timezone = None

    def debug(self, *message):
        self(*message, level='debug')

    def error(self, e, *message):
        if len(message) == 0:
            message = (str(e),)
        if isinstance(e, Exception):
            stack = ''.join(traceback.format_exception(None, e, e.__traceback__))
        else:
            stack = ''
        self(*message, level='error', stack=stack)

    def info(self, *message):
        self(*message, level='info')

    def set_trace_id(self, id):
        self.context.trace_id = id

    def set_req_id(self, id):
        self.context.req_id = id

    def warn(self, *message):
        self(*message, level='warn')

    @property
    def trace_id(self):
        return self.context.trace_id

    @property
    def req_id(self):
        return self.context.req_id


def create_for_collection(action, trace_id, req_id, cost, theme='stable_diffusion'):
    '''
    print message for Big data collection.
    '''
    return json.dumps(
        {
            "business_type": "A1",
            "theme": theme,
            "action": action,
            "distinct_id": str(uuid.uuid4()),
            "time": int(time.time() * 1000),  # ms
            "properties": {
                "cost": int(cost * 1000),  # ms
                "traceId": trace_id,
                "reqId": req_id,
            },
        }
    )


class statistical_runtime:
    context = threading.local()
    uploaded_collection = {
        'load_model': functools.partial(create_for_collection, 'load_model', ),
        'inference': functools.partial(create_for_collection, 'inference', ),
    }

    def __init__(self, name=None, formula=lambda start, end: end - start, unit='s', collection_key=None):
        self.name = name
        self.formula = formula
        self.unit = unit
        self.collection_key = collection_key

    @classmethod
    def reset_collection(cls):
        cls.context._collection = {}

    @classmethod
    def get_collection(cls):
        if hasattr(cls.context, '_collection'):
            return cls.context._collection

    def __call__(self, func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            collection = self.get_collection()
            assert collection is not None, 'The collection needs to be initialized'
            start = time.time()
            result = func(*args, **kwargs)
            key = func.__name__ if self.name is None else self.name
            with TRACER.start_as_current_span(key) as span:
                cost = self.cost(start)
                spend = self.format(cost)
                span: Span
                span.add_event(key, {'costs': spend})
                METER_COUNTER.add(1, {key: cost})
            if self.collection_key is not None and self.collection_key in self.uploaded_collection:
                logger.info(self.uploaded_collection[self.collection_key](logger.trace_id, logger.req_id, cost))
            logger.info(f'{key} costs {spend}')
            self.get_collection()[key] = spend
            return result

        return inner

    def __enter__(self):
        collection = self.get_collection()
        assert collection is not None, 'The collection needs to be initialized'
        assert self.name is not None, 'name is required'
        self.span: Span = TRACER.start_span(self.name)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        cost = self.cost(self.start)
        spend = self.format(cost)
        if self.collection_key is not None and self.collection_key in self.uploaded_collection:
            logger.info(self.uploaded_collection[self.collection_key](logger.trace_id, logger.req_id, cost))
        logger.info(f'{self.name} costs {spend}')
        self.get_collection()[self.name] = spend
        self.span.add_event(self.name, {'costs': spend})
        METER_COUNTER.add(1, {self.name: cost})
        self.span.end()

    def cost(self, start):
        return round(self.formula(start, time.time()), 2)

    def format(self, cost):
        spend = f"{cost} {self.unit}"
        return spend


logger = Logger()
