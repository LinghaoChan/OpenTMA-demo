from functools import partial
import os
import uvicorn
import torch
import numpy as np
import gradio as gr
import gdown
import time

from load import load_model, load_json
from load import load_unit_motion_embs, load_keyids_splits


WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'> Whole-body Motion Retrieval Model </h1>
<br> The whole-body motion retrieval model is used in <a href="https://lhchen.top/HumanTOMATO/">HumanTOMATO</a> project. The demo space is powered by <a href="https://swanhub.co/">SwanHub</a> platform. 
</div>
"""

EXAMPLES = [
    "the guy is performing kung fu",
    "tai chi",
    "a boy is doing tai chi",
    "dance break",
    "Strange Weapon Nebelschlag",
    "a girl is performing Strange Weapon Nebelschlag",
    "a person bends down forward",
    "A man is playing the piano",
    "A baby is jumping",
    "Plays the violin sadly",
    "A person is swimming",
    "intense entertainment battle.",
    "walked straight to end then turned and waked to end",
    "erhu",
    "Someone walks backward",
    "Somebody is ascending a staircase",
    "dances Middle Hip-hop Popcorn",
    "play big ruan",
    "a person walking slowly forward with confidence",
    "sitting while drying the quilt",
    "The person walked forward and is picking up his toolbox"
]

# Show closest text in the training


# css to make videos look nice
# var(--block-border-color);
CSS = """
.retrieved_video {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 100%;
    line-height: var(--line-sm);
}

.contour_video {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: var(--layer-5);
    border-radius: var(--block-radius);
    background: var(--background-fill-primary);
    padding: 0 var(--size-6);
    max-height: var(--size-screen-h);
    overflow: hidden;
}
"""


DEFAULT_TEXT = "A person is "

def humanml3d_keyid_to_babel_rendered_url(h3d_index, amass_to_babel, keyid):
    # Don't show the mirrored version of HumanMl3D
    if "M" in keyid:
        return None

    dico = h3d_index[keyid]
    path = dico["path"]

    # HumanAct12 motions are not rendered online
    # so we skip them for now
    if "humanact12" in path:
        return None

    # This motion is not rendered in BABEL
    # so we skip them for now
    if path not in amass_to_babel:
        return None

    babel_id = amass_to_babel[path].zfill(6)
    url = f"https://babel-renders.s3.eu-central-1.amazonaws.com/{babel_id}.mp4"

    # For the demo, we retrieve from the first annotation only
    ann = dico["annotations"][0]
    start = ann["start"]
    end = ann["end"]
    text = ann["text"]

    data = {
        "url": url,
        "start": start,
        "end": end,
        "tšxt": text,
        "keyid": keyid,
        "babel_id": babel_id,
        "path": path
    }

    return data


def retrieve(model, all_names, keyid_to_url, all_unit_motion_embs, all_keyids, text, splits="test", nmax=8):
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # import pdb; pdb.set_trace()
    # unit_motion_embs = torch.cat([all_unit_motion_embs[s] for s in splits])
    unit_motion_embs = all_unit_motion_embs[splits].to(model.device)
    print(f"Retrieval set ({splits}) size:", unit_motion_embs.shape)
    # keyids = np.concatenate([all_keyids[s] for s in splits])
    # idlist = [i for i in range(len(unit_motion_embs))]
    keyids = np.array(all_keyids[splits])

    scores = model.compute_scores(text, unit_embs=unit_motion_embs)

    sorted_idxs = np.argsort(-scores)
    best_keyids = keyids[sorted_idxs]
    best_scores = scores[sorted_idxs]
    # import pdb; pdb.set_trace()
    best_names = np.array(all_names[splits])[sorted_idxs]

    datas = []
    print("============================")
    print("Input text:", text)
    print("============================")
    for keyid, score, name in zip(best_keyids, best_scores, best_names):
        if len(datas) == nmax:
            break
        idstr = keyid
        
        mp4dir = f"https://motionx.deepdataspace.com/visualization/{idstr}.mp4"
        print("Motion ID: "+idstr, "; Caption", name)
        data = {
            "mp4dir": mp4dir,
            "text": name,
            "idname": idstr
        }
        # data = keyid_to_url(keyid)
        if data is None:
            continue
        data["score"] = round(float(score), 2)
        datas.append(data)
    return datas


# HTML component
def get_video_html(data, video_id, width=700, height=700):
    text = data["text"]
    score = data["score"]
    mp4dir = data["mp4dir"]
    idname = data["idname"]
    # url = data["url"]
    # start = data["start"]
    # end = data["end"]
    # score = data["score"]
    # text = data["text"]
    # keyid = data["keyid"]
    # babel_id = data["babel_id"]
    # path = data["path"]

    # trim = f"#t={start},{end}"
    title = f'''Score = {score};
MP4ID: {idname};
Corresponding text: {text}'''

    # class="wrap default svelte-gjihhp hide"
    # <div class="contour_video" style="position: absolute; padding: 10px;">
    # width="{width}" height="{height}"
    video_html = f'''
<video class="retrieved_video" width="{width}" height="{height}" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture id="{video_id}" title="{title}">
  <source src="{mp4dir}" type="video/mp4">
  Your browser does not support the video tag.
</video>
'''
    return video_html


def retrieve_component(retrieve_function, text, splits_choice, nvids, n_component=24):
    if text == DEFAULT_TEXT or text == "" or text is None:
        return [None for _ in range(n_component)]

    # cannot produce more than n_compoenent
    nvids = min(nvids, n_component)

    if "Unseen" in splits_choice:
        splits = "test"
    else:
        splits = "all"

    datas = retrieve_function(text, splits=splits, nmax=nvids)
    htmls = [get_video_html(data, idx) for idx, data in enumerate(datas)]
    # get n_component exactly if asked less
    # pad with dummy blocks
    htmls = htmls + [None for _ in range(max(0, n_component-nvids))]
    return htmls


# if not os.path.exists("data"):
#     gdown.download_folder("https://drive.google.com/drive/folders/1MgPFgHZ28AMd01M1tJ7YW_1-ut3-4j08",
#                           use_cookies=False)

test_dir = "/comp_robot/lushunlin/visualization/visualization/output/test_motion"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# LOADING
from pathlib import Path
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)
# app.mount("/static", StaticFiles(directory=static_dir), name="static")
model = load_model(device)
# import pdb; pdb.set_trace()
# splits = ["train", "val", "test"]
splits = ["test"]
all_unit_motion_embs = load_unit_motion_embs(device)
# import pdb; pdb.set_trace()
# all_keyids = load_keyids_splits(splits)
all_keyids = []
with open('./embeddings/names.txt', 'r', encoding='utf-8') as f:
    for ann in f.readlines():
        ann = ann.strip('\n')
        all_keyids.append(ann)
all_names = []
with open('./embeddings/caption.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
        name = name.strip('\n')
        all_names.append(name)
TEST_keyids = []
with open('./TEST_embeddings/names.txt', 'r', encoding='utf-8') as f:
    for ann in f.readlines():
        ann = ann.strip('\n')
        TEST_keyids.append(ann)
TEST_names = []
with open('./TEST_embeddings/caption.txt', 'r', encoding='utf-8') as f:
    for name in f.readlines():
        name = name.strip('\n')
        TEST_names.append(name)
keyids = {"all": all_keyids, "test": TEST_keyids}
names = {"all": all_names, "test": TEST_names}


h3d_index = load_json("amass-annotations/humanml3d.json")
amass_to_babel = load_json("amass-annotations/amass_to_babel.json")

keyid_to_url = partial(humanml3d_keyid_to_babel_rendered_url, h3d_index, amass_to_babel)
# retrieve_function = partial(retrieve, model, all_names, keyid_to_url, all_unit_motion_embs, all_keyids)
retrieve_function = partial(retrieve, model, names, keyid_to_url, all_unit_motion_embs, keyids)

# DEMO
theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
retrieve_and_show = partial(retrieve_component, retrieve_function)

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(scale=2):
                text = gr.Textbox(placeholder="Type the motion you want to search with a sentence",
                                  show_label=True, label="Text prompt", value=DEFAULT_TEXT)
            with gr.Column(scale=1):
                btn = gr.Button("Retrieve", variant='primary')
                clear = gr.Button(value="Clear", variant='secondary')

            with gr.Row():
                with gr.Column(scale=1):
                    splits_choice = gr.Radio(["All motions", "Unseen motions"], label="The motions database is the Motion-X whole-body motion dataset.",
                                             value="Unseen motions",
                                             info="The retrieval model supports two modes: (1) Motion-X full set (77,743 samples), and (2) Motion-X test set (11,752 samples).")

                with gr.Column(scale=1):
                    # nvideo_slider = gr.Slider(minimum=4, maximum=24, step=4, value=8, label="Number of videos")
                    nvideo_slider = gr.Radio([4, 8, 12, 16, 24], label="Videos",
                                             value=8,
                                             info="Number of videos to display")

        with gr.Column(scale=2):
            def retrieve_example(text, splits_choice, nvideo_slider):
                return retrieve_and_show(text, splits_choice, nvideo_slider)

            examples = gr.Examples(examples=[[x, None, None] for x in EXAMPLES],
                                   inputs=[text, splits_choice, nvideo_slider],
                                   examples_per_page=22,
                                   run_on_click=False, cache_examples=False,
                                   fn=retrieve_example, outputs=[])

    i = -1
    # should indent
    for _ in range(6):
        with gr.Row():
            for _ in range(4):
                i += 1
                video = gr.HTML()
                videos.append(video)

    # connect the examples to the output
    # a bit hacky
    examples.outputs = videos

    def load_example(example_id):
        processed_example = examples.non_none_processed_examples[example_id]
        return gr.utils.resolve_singleton(processed_example)

    examples.dataset.click(
        load_example,
        inputs=[examples.dataset],
        outputs=examples.inputs_with_examples,  # type: ignore
        show_progress=False,
        postprocess=False,
        queue=False,
        ).then(
            fn=retrieve_example,
            inputs=examples.inputs,
            outputs=videos
        )

    btn.click(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)
    text.submit(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)
    splits_choice.change(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)
    nvideo_slider.change(fn=retrieve_and_show, inputs=[text, splits_choice, nvideo_slider], outputs=videos)

    def clear_videos():
        return [None for x in range(24)] + [DEFAULT_TEXT]

    clear.click(fn=clear_videos, outputs=videos + [text])
# app = gr.mount_gradio_app(app, demo, path="/")
# demo.launch(share=True)
demo.launch()
# serve the app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=7860)
