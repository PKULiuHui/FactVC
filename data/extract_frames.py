import os
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video frames at 1fps')
    parser.add_argument('--dataset', type=str, default='activitynet', choices=['activitynet', 'youcook2'])
    args = parser.parse_args()

    if not os.path.exists(f'{args.dataset}/frames'):
        os.mkdir(f'{args.dataset}/frames')

    for fn in tqdm(os.listdir(f'{args.dataset}/videos')):
        vid = fn.split('.')[0]
        os.mkdir(f'{args.dataset}/frames/{vid}')
        os.system(f'ffmpeg -i {args.dataset}/videos/{fn} -vf fps=1 -q:v 2 -start_number 0 {args.dataset}/frames/{vid}/%d.jpg')

