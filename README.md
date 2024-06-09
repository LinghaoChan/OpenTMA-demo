This is a demo repository for OpenTMA. The purpose of this repository is to demonstrate the capabilities of [OpenTMA](https://github.com/LinghaoChan/OpenTMA). 

❓ How to use this repository?

1. Clone this repository to your local machine.

```bash
git clone https://github.com/LinghaoChan/OpenTMA-demo.git
```

2. Install OpenTMA.

```bash
pip install -r requirements.txt
```

3. Download the pre-trained model and the motion/text/sbert embeddings from the [Google Drive](https://drive.google.com/file/d/1LHvonwW8JWKpcwj3SyF0cA83wIIBNX_m/view?usp=sharing). Unzip the file and put the `embeddings` folder and the `textencoder.ckpt` in the root directory of this repository. Your file tree should look like this:

```bash
OpenTMA-demo/
├── amass-annotations
├── app.py
├── embeddings
│   ├── caption.txt
│   ├── motion_embedding.npy
│   ├── names.txt
│   ├── sbert_embedding.npy
│   └── text_embedding.npy
├── load.py
├── mld
├── model.py
├── README.md
├── requirements.txt
├── TEST_embeddings
│   ├── caption.txt
│   ├── motion_embedding.npy
│   ├── names.txt
│   ├── sbert_embedding.npy
│   └── text_embedding.npy
├── test_temos.py
└── textencoder.ckpt
```

4. Run the demo.
```bash
python app.py
```

We also deploy our demo on [SwanHub](https://swanhub.co/Evan/OpenTMR/tree/master), see [demo](https://swanhub.co/demo/Evan/OpenTMR). 

For the details of the project and the citation, please refer to the [HumanTOMATO](https://github.com/IDEA-Research/HumanTOMATO) project.
