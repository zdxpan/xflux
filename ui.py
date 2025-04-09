import argparse
import gradio as gr
import re
import traceback

from dispatcher                     import Dispatcher
from log                            import logger
from multidict                      import MultiDict
from PIL                            import Image, ImageOps
from tensor_util                    import RESIZER

from flux_pipe import FluxPipeline, FluxNormalInput, normalize_size
from text_encoder import PromptEncoder

def convert_mask(image):
    mask = Image.new(mode='L', size=image.size, color=255)
    mask.putdata(image.getdata(band=3))
    return mask


def create_ui(dispatcher):
    
    def generate_image(
        prompt, image, strength, blur,
        height, width, resizer, base_shift, max_shift, steps, guidance,
        seed, variant,
        do_true_cfg, cfg_zero_init, cfg_skip
    ):
        try:
            file = []
            args = MultiDict([
                ('prompt', prompt),
                ('strength', strength),
                ('height', height),
                ('width', width),
                ('resizer', resizer),
                ('base_shift', base_shift),
                ('max_shift', max_shift),
                ('steps', steps),
                ('guidance', guidance),
                ('seed', seed),
                ('variant', variant),
                ('do_true_cfg', do_true_cfg),
                ('use_zero_init', cfg_zero_init),
                ('cfg_skip', cfg_skip),
            ])

            background = Image.open(image['background']['path'])
            mask = Image.open(image['layers'][0]['path'])
            #mask = Image.open('test/car_mask.png').convert('RGB')
            if not is_empty_image(background):
                args.add('file', background)
                args.add('image', 0)
                if not is_empty_image(mask):
                    args.add('file', convert_mask(mask))
                    #args.add('file', mask)
                    args.add('mask', 1)

            result = dispatcher('generate', args)

            images = []
            for i in range(len(result['images'])):
                image = result['images'][i]
                if not 'nsfw' in result or not result['nsfw'][i]:
                    images.append(image)
                result['images'][i] = f'<ImageFile size={image.size[0]}x{image.size[1]}>'
        except Exception as e:
            traceback.print_exc()
            images = None
            result = {
                'exception': str(e)
            }
        finally:
            return [ images, result ]

    def load_from_gallery(gallery, index=0):
        image = None
        if gallery is not None and len(gallery) > index:
            image = {
                'background': Image.open(gallery[index][0]),
                'layers': None,
                'composite': None
            }
        return image

    theme = gr.themes.Base(
        primary_hue='green',
        neutral_hue='neutral'
    ).set(
        slider_color='#FF3333',
        checkbox_border_color_selected='#17A34A',
        checkbox_background_color_selected='#17A34A'
    )
    css = ' '.join([
        '.no-grow {flex-grow:0 !important;}'
    ])
    block = gr.Blocks(theme=theme, css=css, analytics_enabled=False).queue()
    with block:
        with gr.Row():
            gr.HTML(
                '<h3 style="margin:0"> 🎡 <span style="color:#FF3333">PLAY</span><span style="color:#559955">GROUND</span><sup>FX</sup></h3>'
            )
        with gr.Row():
            with gr.Column(scale=4, variant='panel'):
                gallery = gr.Gallery(
                    label='Gallery', height=512, preview=True
                )
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder='Prompt', lines=3, max_lines=3, show_label=False, scale=9
                    )
                    generate = gr.Button(
                        value='Generate', variant='primary', scale=1, min_width=128
                    )
                meta = gr.JSON(
                    label='Meta', scale=1
                )
            with gr.Column(scale=2):
                with gr.Column(variant='panel', elem_classes='no-grow'):
                    with gr.Group():
                        image = gr.ImageEditor(
                            sources=['upload', 'clipboard'], type='pil', image_mode='RGBA', canvas_size=(1024, 1024),
                            show_label=False, layers=False,
                            brush=gr.Brush(colors=['#000', '#F00', '#0F0', '#00F'], color_mode='fixed')
                        )
                        with gr.Accordion(label='Extra images..', open=False):
                            extra_images = gr.Gallery(
                                container=False, height=240
                            )
                    with gr.Row():
                        load_image = gr.Button(
                            value='Load', size='sm', scale=1, min_width=64
                        )
                    with gr.Row():
                        strength = gr.Slider(
                            label='Strength', minimum=0.0, maximum=1.0, value=0.75, step=0.05
                        )
                        blur = gr.Slider(
                            label='Mask Blur', minimum=0.0, maximum=1.0, value=0.0, step=0.05
                        )
                    with gr.Group():
                        with gr.Row():
                            height = gr.Slider(
                                label='Height', minimum=512, maximum=4096, value=1024, step=64
                            )
                            width = gr.Slider(
                                label='Width', minimum=512, maximum=4096, value=1024, step=64
                            )
                            resizer = gr.Dropdown(
                                RESIZER, label='Resize Method', value=RESIZER[0]
                            )
                    with gr.Row():
                        base_shift = gr.Slider(
                            label='Base Shift', minimum=0.0, maximum=10.0, value=0.5, step=0.05
                        )
                        max_shift = gr.Slider(
                            label='Max Shift', minimum=0.0, maximum=10.0, value=1.15, step=0.05
                        )
                    with gr.Row():
                        steps = gr.Slider(
                            label='Steps', minimum=1, maximum=50, value=28, step=1
                        )
                        guidance = gr.Slider(
                            label='Guidance', minimum=0.0, maximum=16.0, value=3.5, step=0.5
                        )
                    with gr.Row():
                        seed = gr.Slider(
                            label='Seed', minimum=-1, maximum=2**32 - 1, value=-1, step=1
                        )
                        variant = gr.Slider(
                            label='Variant', minimum=0.0, maximum=1.0, value=0.0, step=0.01
                        )
                    with gr.Row():
                        do_true_cfg = gr.Checkbox(label='DoTrue cfg', value=False)
                        cfg_zero_init = gr.Checkbox(label='cfg zero', value=False)
                        cfg_skip =  gr.Slider(
                            label='cfg_skip', minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )


        generate.click(
            generate_image,
            inputs=[prompt, image, strength, blur,
                height, width, resizer, base_shift, max_shift, steps, guidance,
                seed, variant, 
                do_true_cfg, cfg_zero_init, cfg_skip
                ],
            outputs=[gallery, meta],
            preprocess=False
        )
        load_image.click(
            load_from_gallery,
            inputs=[gallery],
            outputs=[image],
            show_progress='hidden'
        )
    return block


def is_empty_image(image):
    if image.mode == 'RGBA':
        extrema = image.getextrema()
        if extrema[3][1] == 0:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log-format',
        type=str,
        choices=[ 'json', 'plain' ],
        default='plain',
        help='log format'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=[ 'debug', 'info', 'warn', 'error' ],
        default='debug',
        help='log level'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/FLUX.1-dev-hf',
        help='path to the model directory'
    )
    parser.add_argument(
        '--parallelism',
        type=int,
        default=4,
        help='diffusion execute parallelism'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='port to listen'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=[ 'full', 'half' ],
        default='half',
        help='valuate at this precision'
    )
    parser.add_argument(
        '--timezone',
        type=str,
        default=None,
        help='time zone of log timestamp, format: (GMT|UTC)(+|-)(HH)'
    )

    opt = parser.parse_args()

    if opt.timezone != None:
        tz = re.search(r'^(GMT|UTC)(\+|\-)(\d+)$', opt.timezone)
        if tz == None:
            raise argparse.ArgumentTypeError(
                f'argument --timezone: invalid value: \'{opt.timezone}\''
            )
        opt.timezone = int(tz.group(2) + tz.group(3))

    return opt


if __name__ == '__main__':
    opt = parse_args()

    logger.config(
        format=opt.log_format,
        level=opt.log_level,
        timezone=opt.timezone
    )

    dispatcher = Dispatcher(opt)

    # Start gradio app
    block = create_ui(dispatcher)
    block.launch(server_name='0.0.0.0', server_port=opt.port, show_api=False)

    # Also stopped consumer threads after flask app exited
    dispatcher.stop()
