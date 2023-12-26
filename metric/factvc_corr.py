import re
import json
import argparse
import numpy as np
import torch
import clip
import math
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

from emscore.scorer import EMScorer
from emscore.utils import get_idf_dict


def get_anno_score(dataset):
    """Obtain factuality annotation scores (paragraph/sentence/word-level scores)"""
    vids = [line.strip() for line in open(f'../data/{dataset}/vids.txt')]
    anno = json.load(open(f'../data/{dataset}/factuality_annotation.json'))
    if dataset == 'activitynet':
        model_names = ['MART', 'COOT', 'PDVC_gt', 'PDVC_pred', 'Song', 'VLTinT']
    else:
        model_names = ['VTrans', 'MART', 'COOT', 'COOT_100m', 'UniVL', 'VLTinT']
    model_scores = [{'word_score': [], 'sent_score': [], 'para_score': []} for _ in range(6)]
    for vid in vids:
        for i in range(6):
            cur_anno = anno[vid][model_names[i]]
            cur_para_score = cur_anno['paragraph_score']
            cur_sent_score = .0
            cur_span_len = .0
            cur_total_len = .0
            for sent in cur_anno['sentences']:
                span_l, span_r = [], []
                sent = sent.replace('.', '').strip()
                for j, char in enumerate(sent):
                    if char == '[':
                        span_l.append(j)
                    elif char == ']':
                        span_r.append(j)
                assert len(span_l) == len(span_r), vid
                if len(span_l) > 0:
                    cur_sent_score += 1.0
                cur_total_len += len(sent.split())
                for l, r in zip(span_l, span_r):
                    cur_span_len += len(sent[l:r].split())
            cur_sent_score /= len(cur_anno['sentences'])
            cur_word_score = cur_span_len / cur_total_len
            model_scores[i]['para_score'].append(cur_para_score)
            model_scores[i]['sent_score'].append(cur_sent_score)
            model_scores[i]['word_score'].append(cur_word_score)

    return model_scores


def parse_sent(sent):
    res = re.sub('[^a-zA-Z]', ' ', sent)
    res = res.strip().lower().split()
    return res


def get_caption(dataset):
    """Obtain ground-truth captions and model-generated captions"""
    vids = [line.strip() for line in open(f'../data/{dataset}/vids.txt')]
    gt_caption, model_caption = [], []
    if dataset == 'activitynet':
        gt_cap_files = [f'../data/{dataset}/captions/gt_ae_test_1_para.json',
                        f'../data/{dataset}/captions/gt_ae_test_2_para.json']
        gt_cap_jsons = [json.load(open(fn)) for fn in gt_cap_files]
        for vid in vids:
            cur_refs = [' '.join(parse_sent(gt_cap_jsons[0][vid]))]
            if vid in gt_cap_jsons[1]:
                cur_refs.append(' '.join(parse_sent(gt_cap_jsons[1][vid])))
            else:
                cur_refs.append(' '.join(parse_sent(gt_cap_jsons[0][vid])))
            gt_caption.append(cur_refs)
        model_cap_files = [f'../data/{dataset}/captions/mart_ae_test.json',
                           f'../data/{dataset}/captions/coot_ae_test.json',
                           f'../data/{dataset}/captions/pdvc_gt_ae_test.json',
                           f'../data/{dataset}/captions/pdvc_pred_ae_test.json',
                           f'../data/{dataset}/captions/song_ae_test.json',
                           f'../data/{dataset}/captions/vltint_ae_test.json']
        model_cap_jsons = [json.load(open(fn))['results'] for fn in model_cap_files]
        model_caption = [[] for _ in range(6)]
        for vid in vids:
            for i in range(6):
                sents = [x['sentence'].strip() for x in model_cap_jsons[i][vid]]
                model_caption[i].append(' '.join(parse_sent(' '.join(sents))))

    else:
        gt_cap_json = json.load(open(f'../data/{dataset}/captions/gt_val_para.json'))
        for vid in vids:
            gt_caption.append([' '.join(parse_sent(gt_cap_json[vid]))])
        model_cap_files = [f'../data/{dataset}/captions/vtrans_val.json',
                           f'../data/{dataset}/captions/mart_val.json',
                           f'../data/{dataset}/captions/coot_val.json',
                           f'../data/{dataset}/captions/coot_100m_val.json',
                           f'../data/{dataset}/captions/univl_val.json',
                           f'../data/{dataset}/captions/vltint_val.json']
        model_cap_jsons = [json.load(open(fn))['results'] for fn in model_cap_files]
        model_caption = [[] for _ in range(6)]
        for vid in vids:
            for i in range(6):
                sents = [x['sentence'].strip() for x in model_cap_jsons[i][vid]]
                model_caption[i].append(' '.join(parse_sent(' '.join(sents))))

    gt_caption_flatten = gt_caption * 6
    model_caption_flatten = []
    for cap_list in model_caption:
        model_caption_flatten.extend(cap_list)

    return gt_caption_flatten, model_caption_flatten


def get_factvc_score(dataset, gt_caption, model_caption, pre_load_vid=False):
    """Compute FactVC scores"""
    clip_model = '../pretrained_models/factvc_video.pth'
    print(f'Using clip model {clip_model}')
    vids = [line.strip() for line in open(f'../data/{dataset}/vids.txt')]
    if pre_load_vid:
        vid_feat_dict = {vid: torch.from_numpy(np.load(f'clip_feat_factvc/{vid}.npy')) for vid in vids}
    else:
        print('Extracting video features...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(clip_model, device=device)
        frame_dir = f'../data/{dataset}/frames'
        vid_feat_dict = {}
        for vid in tqdm(vids):
            cur_dir = f'{frame_dir}/{vid}'
            images = []
            for i in range(0, len(os.listdir(cur_dir)), 1):
                img_path = f'{cur_dir}/{i}.jpg'
                images.append(preprocess(Image.open(img_path)))
            image_input = torch.tensor(np.stack(images)).cuda()
            image_features_list = []
            bs = 256
            with torch.no_grad():
                n_inter = math.ceil(len(image_input) / bs)
                for i in range(n_inter):
                    image_features = model.encode_image(image_input[i * bs: (i + 1) * bs]).float()
                    image_features_list.append(image_features)
            image_features = torch.cat(image_features_list, dim=0)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            vid_feat_dict[vid] = image_features.cpu()
            # np.save(f'clip_feat_factvc/{vid}.npy', vid_feat_dict[vid].numpy())
    vids_flatten = vids * 6

    # prepare idf
    corpus_json = json.load(open(f'../data/{dataset}/captions/ref_paragraphs.json'))
    corpus = []
    for v_id, sent in corpus_json.items():
        corpus.append(' '.join(parse_sent(sent[0])))
    idf_dict = get_idf_dict(corpus, clip.tokenize, nthreads=4)

    metric = EMScorer(vid_feat_cache=vid_feat_dict, clip_model=clip_model)
    results = metric.score(cands=model_caption, refs=gt_caption, vids=vids_flatten, idf=idf_dict,
                           cogr_weight=0.25, ref_weight=0.5, batch_size=64)

    return results


def get_corr(anno_score, metric_score):
    """Compute correlation between FactVC and annotation scores"""
    all_anno_score = {'para_score': [], 'sent_score': [], 'word_score': []}
    all_metric_score = {'V': [], 'T': [], 'VT': []}
    for i in range(6):
        for m in all_anno_score:
            all_anno_score[m] += anno_score[i][m]
    all_metric_score['V'] = metric_score['EMScore(X,V)']['full_P']
    all_metric_score['T'] = metric_score['EMScore(X,X*)']['full_P']
    all_metric_score['VT'] = metric_score['EMScore(X,V,X*)']['full_P']

    data = pd.DataFrame({
        'word': - np.array(all_anno_score['word_score']),
        'sent': - np.array(all_anno_score['sent_score']),
        'para': np.array(all_anno_score['para_score']),
        'V': np.array(all_metric_score['V']),
        'T': np.array(all_metric_score['T']),
        'VT': np.array(all_metric_score['VT']),
    })
    print('Pearson corr between auto metric and factual anno:')
    for m in all_metric_score:
        print(m, data.corr()[m]['para'], data.corr()[m]['sent'], data.corr()[m]['word'])

    return data.corr()['V']['para'], data.corr()['V']['sent'], data.corr()['V']['word'], \
           data.corr()['T']['para'], data.corr()['T']['sent'], data.corr()['T']['word'], \
           data.corr()['VT']['para'], data.corr()['VT']['sent'], data.corr()['VT']['word']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute correlation between FactVC and factuality annotation')
    parser.add_argument('--dataset', type=str, default='activitynet', choices=['activitynet', 'youcook2'])
    args = parser.parse_args()

    anno_score = get_anno_score(args.dataset)
    gt_cap, model_cap = get_caption(args.dataset)
    factvc_score = get_factvc_score(args.dataset, gt_cap, model_cap)
    get_corr(anno_score, factvc_score)
