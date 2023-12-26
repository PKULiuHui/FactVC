# FactVC: Factual consistency for Video Captioning

This repository contains the data and code for the paper "Models See Hallucinations: Evaluating the Factuality in Video Captioning".

## File Structure

```
FactVC-main/
├── data/
│   ├── activitynet/
│   │   ├── videos/         # sampled ActivityNet videos
│   │   ├── frames/         # extracted video frames
│   │   ├── captions/       # ground-truth and model-generated captions
│   │   ├── vids.txt        # video ids
│   │   └── factuality_annotation.json  # human factuality annotation
│   ├── youcook2/
│   │   ├── videos/         # sampled YouCook2 videos
│   │   ├── frames/         # extracted video frames
│   │   ├── captions/       # ground-truth and model-generated captions
│   │   ├── vids.txt        # video ids
│   │   └── factuality_annotation.json  # human factuality annotation
│   └── extract_frames.py
├── metric/
│   ├── clip/
│   ├── emscore/
│   └── factvc_corr.py      # code to compute FactVC score and correlation
└── pretrained_models
    └── factvc_video.pth    # our pretrained metric model
```

## Usage

First, download the sampled [ActivityNet videos](https://drive.google.com/file/d/1-92SRIyLK2tjC-8u-cXqq5KuF7Dq8vfd/view?usp=drive_link) and [YouCook2 videos](https://drive.google.com/file/d/1-3wOvb3ft4vwyrieq3k3P4-geK05_iGG/view?usp=drive_link) and unzip them into corresponding folders. Download the pretrained [FactVC metric model](https://drive.google.com/file/d/1S9T4-XLHMhRt3NW4NRQ3WflmEaEZ_H5I/view?usp=drive_link) and put it under ```pretrained_models/``` folder.

Then, extract video frames at 1fps (used for computing FactVC metric scores):

```bash
cd data/
python extract_frames.py --dataset activitynet
python extract_frames.py --dataset youcook2
```

Now, you can compute the FactVC scores and the correlation between FactVC score and human annotation:

```bash
cd metric/
python factvc_corr.py --dataset activitynet
python factvc_corr.py --dataset youcook2
```



## Acknowledgements

We acknowledge the [EMScore project](https://github.com/ShiYaya/emscore) that we based on our work: