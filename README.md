# LayerAnimate: Layer-level Control for Animation

[Yuxue Yang](https://yuxueyang1204.github.io/)<sup>1,2</sup>, [Lue Fan](https://lue.fan/)<sup>2</sup>, [Zuzeng Lin](https://www.researchgate.net/scientific-contributions/Zuzeng-Lin-2192777418)<sup>3</sup>, [Feng Wang](https://happynear.wang/)<sup>4</sup>, [Zhaoxiang Zhang](https://zhaoxiangzhang.net)<sup>1,2†</sup>

<sup>1</sup>UCAS&emsp; <sup>2</sup>CASIA&emsp; <sup>3</sup>TJU&emsp; <sup>4</sup>CreateAI&emsp; <sup>†</sup>Corresponding author

<a href='https://arxiv.org/abs/2501.08295'><img src='https://img.shields.io/badge/arXiv-2501.08295-b31b1b.svg'></a> &nbsp;
<a href='https://layeranimate.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://www.bilibili.com/video/BV1EycqeaEqF/'><img src='https://img.shields.io/badge/BiliBili-Video-479fd1.svg'></a> &nbsp;
<a href='https://youtu.be/b_bvVKigky4'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a> &nbsp;
<a href='https://huggingface.co/spaces/IamCreateAI/LayerAnimate'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-Demo-blue'></a><br>

Official implementation of **LayerAnimate: Layer-level Control for Animation**, ICCV 2025

<div align="center"> <img src='__assets__/figs/demos.gif'></img></div>

**Videos on the [project website](https://layeranimate.github.io) vividly introduces our work and presents qualitative results for an enhanced view experience.**

## Updates

- [25-06-26] Our work is accepted by ICCV 2025! 🎉
- [25-05-29] We have extended LayerAnimate to the DiT ([Wan2.1 1.3B](https://github.com/Wan-Video/Wan2.1)) variant, enabling the generation of 81 frames at 480 × 832 resolution. It performs surprisingly well in the [Real-World Domain](https://layeranimate.github.io/#real_world) shown in the project website.
- [25-03-31] Release the online demo on [Hugging Face](https://huggingface.co/spaces/IamCreateAI/LayerAnimate).
- [25-03-30] Release a gradio script [app.py](scripts/app.py) to run the demo locally. Please raise an issue if you encounter any problems.
- [25-03-22] Release the checkpoint and the inference script. **We update layer curation pipeline and support trajectory control for a flexible composition of various layer-level controls.**
- [25-01-15] Release the project page and the arXiv preprint.

## Installation

```bash
git clone git@github.com:IamCreateAI/LayerAnimate.git
conda create -n layeranimate python=3.10 -y
conda activate layeranimate
pip install -r requirements.txt
pip install wan@git+https://github.com/Wan-Video/Wan2.1  # If you want to use DiT variant.
```

## Models

| Models                   | Download Link                                                                                                                                           | Video Size        |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| UNet variant | [Huggingface](https://huggingface.co/Yuppie1204/LayerAnimate-Mix) 🤗  | 16 x 320 x 512  |
| DiT variant | [Huggingface](https://huggingface.co/Yuppie1204/LayerAnimate-DiT) 🤗  | 81 x 480 x 832  |

Download the pretrained weights and put them in `checkpoints/` directory as follows:

```bash
checkpoints/
├─ LayerAnimate-Mix (UNet variant)
└─ LayerAnimate-DiT
```

## Inference script

### UNet variant (Paper version)

Run the following command to generate a video from input images:

```bash
python scripts/animate_Layer.py --config scripts/demo1.yaml --savedir outputs/sample1

python scripts/animate_Layer.py --config scripts/demo2.yaml --savedir outputs/sample2

python scripts/animate_Layer.py --config scripts/demo3.yaml --savedir outputs/sample3

python scripts/animate_Layer.py --config scripts/demo4.yaml --savedir outputs/sample4

python scripts/animate_Layer.py --config scripts/demo5.yaml --savedir outputs/sample5
```

Note that the layer-level controls are prepared in `__assets__/demos`.

#### Run demo locally

You can run the demo locally by executing the following command:

```bash
python scripts/app.py --savedir outputs/gradio
```

Then, open the link in your browser to access the demo interface. The output video and the video with trajectory will be saved in the `outputs/gradio` directory.

### DiT variant (Wan2.1 1.3B)

Run the following command to generate a video from input images:

```bash
python scripts/infer_DiT.py --config __assets__/demos/realworld/config.yaml --savedir outputs/realworld
```

We take the `config.yaml` in `demos/realworld/` as an example. You can also modify the config file to suit your needs.

## Todo

- [x] Release the code and checkpoint of LayerAnimate.
- [x] Upload a gradio script to run the demo locally.
- [x] Create a online demo in the huggingface space.
- [x] DiT-based LayerAnimate.
- [ ] Release checkpoints trained under single control modality with better performance.
- [ ] Release layer curation pipeline.
- [ ] Training script for LayerAnimate.

## Acknowledgements

We sincerely thank the great work [ToonCrafter](https://doubiiu.github.io/projects/ToonCrafter/), [LVCD](https://luckyhzt.github.io/lvcd), [AniDoc](https://yihao-meng.github.io/AniDoc_demo/), and [Wan-Video](https://github.com/Wan-Video/Wan2.1) for their inspiring work and contributions to the AIGC community.

## Citation

Please consider citing our work as follows if it is helpful.
```bib
@article{yang2025layeranimate,
  author    = {Yang, Yuxue and Fan, Lue and Lin, Zuzeng and Wang, Feng and Zhang, Zhaoxiang},
  title     = {LayerAnimate: Layer-level Control for Animation},
  journal   = {arXiv preprint arXiv:2501.08295},
  year      = {2025},
}
```
