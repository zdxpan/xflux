import os
import requests
import io
import copy
import base64
import yaml
import json
import platform
from openai  import OpenAI

# os.getenv("DASHSCOPE_API_KEY")

def get_os_type():
    system = platform.system()
    if system == "Darwin":
        return "macOS"
    elif system == "Linux":
        return "Linux"
    else:
        return "Unknown"


def get_base64_image(image):
    buffered = io.BytesIO()
    image_rgb = copy.deepcopy(image).convert('RGB')
    image_rgb.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

class BaseClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

class QwenClient(BaseClient):
    def __init__(self, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=os.getenv("DASHSCOPE_API_KEY")):
        super().__init__("https://dashscope.aliyuncs.com/compatible-mode/v1", api_key)
        self.client = OpenAI(api_key = api_key, base_url = base_url,)
    
    def send(self, image, prompt="给这张图生成标题以及20个不重复的tag", model="qwen-vl-plus", **kwargs):
        if isinstance(image, list):
            image = image[0] if len(image) > 0 else None
        if image:
            img_str = get_base64_image(image)
        msgs1 = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
        msgs2 = [{"role": "user","content": [
                        {"type": "text","text": prompt},
                        {"type": "image_url",  "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}} if image  else {}
                    ]
                }]
        completion = self.client.chat.completions.create(
            # model="qwen-vl-plus",
            model = model,  messages=msgs1 if 'vl' not in model else msgs2) # type: ignore
        res = completion.model_dump_json()
        out_json = json.loads(res)
        api_response = {
            'code': 200,
            'content': None,
            'msg': 'Chat completed successfully'
        }
        try:
            if 'choices' in out_json and len(out_json['choices']) > 0:
                content = out_json['choices'][0]['message']['content']
                api_response['content'] = content    # str
            else:
                api_response['code'] = 300
            api_response['msg'] = res
        except Exception as e:
            api_response['code'] = 300
            api_response['msg'] = f'An error occurred: {str(e)}'
        return api_response

class MinicpmClient(BaseClient):
    def __init__(self, base_url,  api_key):
        super().__init__(base_url=base_url, api_key='')

    def send(self, image, prompt="给这张图生成标题以及20个不重复的tag", model="", **kwargs):
        api_url = self.base_url
        do_sample = kwargs.get('do_sample', True)
        num_beams = kwargs.get('num_beams', 3)        
        temperature = kwargs.get('temperature', 0.5)
        top_k = kwargs.get('top_k', 70)
        top_p = kwargs.get('top_p', 0.8)
        min_new_tokens = kwargs.get('min_new_tokens', 16)
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        repetition_penalty = kwargs.get('repetition_penalty', 1.02)
        if isinstance(image, list):
            image = image[0]
        if image is None:
            return {'code': 500, 'msg': 'not a valid image'}
        prompt = [prompt]
        data = {
            "trace_id": kwargs.get("trace_id"), "req_id":kwargs.get("req_id"), "family": "minicpm", "prompt": prompt,  "do_sample": True,  "num_beams": num_beams,
            "temperature": temperature,  "top_k": top_k,  "top_p": top_p,
            "min_new_tokens": min_new_tokens, "max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty,
        }
        files = []
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        files.append(
            ('file', buffered, ) # ('file', ('1.png', buffered, 'application/octet-stream'))  # ok pass
        )
        response = requests.post(api_url, data=data, files=files)
        api_response = {
            'code': response.status_code,
            'content': None,
            'msg': 'Chat completed successfully'
        }
        res_json = response.json()
        if response.status_code == 200:
            api_response['content'] = res_json['texts'][0] if 'texts' in res_json and len(res_json['texts']) > 0 else ''
        api_response['msg'] = res_json
        return api_response


if __name__ == '__main__':
    from PIL import Image
    import yaml
    with open('../config.yaml', "r") as f:
        config = yaml.safe_load(f)
    
    configs = {
        'qwen': { 'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1", 'api_key': os.getenv("DASHSCOPE_API_KEY") },
        # 'minicpm': { 'base_url': "http://prod.tair.a1-llm.xiaopiu.com//api/llava/generate", 'api_key': '' }
        'minicpm': { 'base_url': "http://192.168.1.4:5001/api/minicpm/generate", 'api_key': '' }        # ok pass    
    }
    models = {
        'qwen': QwenClient,
        'minicpm': MinicpmClient
    }
    model_name = config.get('llm_model')
    base_url = config.get('base_url')
    api_key = config.get('api_key')
    client = models.get(model_name)()

    image_path1 = '/home/dell/workspace/js-flux-svc/img/test_call_flux.jpeg'
    image_path2 = '/Users/zdx/Desktop/中国女性，白粉红玫瑰_base_ref_style_02_attnW_05.png'
    os_type = get_os_type()
    image_path = image_path1 if os_type == 'Linux' else image_path2

    image = Image.open(image_path)
    prompt = 'brief describe this image'
    kwargs = {}
    msg = client.send(image=image,prompt=prompt,**kwargs)
    msg

    for model_name in ['qwen', 'minicpm']:
        client = models.get(model_name)(**configs[model_name])
        
        msg = client.send(image=image,prompt=prompt,**kwargs)
        code = msg['code'],
        content = msg['content'],
        msg['msg']
        print(msg)

