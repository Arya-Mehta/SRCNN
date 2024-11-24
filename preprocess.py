import argparse
import os
import h5py
from PIL import ImageFilter, Image
import numpy as np

# Convert Train Set into pre-processed data.
def train(image_dir, output_dir, sub_sample_size=33, stride=14, upscaling=2):
    lr = []
    hr = []

    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
            image_path = os.path.join(image_dir, file)
            high_res = Image.open(image_path).convert('RGB')
            high_res = high_res.resize(((high_res.width // upscaling) * upscaling, (high_res.height // upscaling) * upscaling),
                                       resample=Image.Resampling.BICUBIC)
            low_res = high_res.resize(((high_res.width // upscaling) * upscaling, (high_res.height // upscaling) * upscaling),
                                      resample=Image.Resampling.BICUBIC)
            low_res = low_res.filter(ImageFilter.GaussianBlur(radius=3))
            high_res = np.asarray(high_res, dtype='float') / 255
            low_res = np.asarray(low_res, dtype='float') / 255

            for i in range(0, high_res.shape[0] - sub_sample_size, stride):
                for j in range(0, high_res.shape[1] - sub_sample_size, stride):
                    hr_patch = high_res[i:i+sub_sample_size, j:j+sub_sample_size, :]
                    lr_patch = low_res[i:i+sub_sample_size, j:j+sub_sample_size, :]
                    hr.append(hr_patch)
                    lr.append(lr_patch)
    
    with h5py.File(output_dir, 'w') as f:
        f.create_dataset('hr', data=np.array(hr, dtype='float32'))
        f.create_dataset('lr', data=np.array(lr, dtype='float32'))

# Convert Evaluation Set into pre-processed data.
def eval(image_dir, output_dir, sub_sample_size=33, stride=14, upscaling=2):
    lr = []
    hr = []

    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
            image_path = os.path.join(image_dir, file)
            high_res = Image.open(image_path).convert('RGB')
            high_res = high_res.resize(((high_res.width // upscaling) * upscaling, (high_res.height // upscaling) * upscaling),
                                       resample=Image.Resampling.BICUBIC)
            low_res = high_res.resize(((high_res.width // upscaling) * upscaling, (high_res.height // upscaling) * upscaling),
                                      resample=Image.Resampling.BICUBIC)
            low_res = low_res.filter(ImageFilter.GaussianBlur(radius=2))
            high_res = np.asarray(high_res, dtype='float') / 255
            low_res = np.asarray(low_res, dtype='float') / 255
        

            for i in range(0, high_res.shape[0] - sub_sample_size, stride):
                for j in range(0, high_res.shape[1] - sub_sample_size, stride):
                    hr_patch = high_res[i:i+sub_sample_size, j:j+sub_sample_size, :]
                    lr_patch = low_res[i:i+sub_sample_size, j:j+sub_sample_size, :]
                    hr.append(hr_patch)
                    lr.append(lr_patch)
    
    with h5py.File(output_dir, 'w') as f:
        f.create_dataset('hr', data=np.array(hr, dtype='float32'))
        f.create_dataset('lr', data=np.array(lr, dtype='float32'))

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for training and evaluation.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands: train or eval')

    # Train command
    train_parser = subparsers.add_parser('train', help='Process training images.')
    train_parser.add_argument('image_dir', type=str, help='Directory containing input images for training.')
    train_parser.add_argument('output_dir', type=str, help='Output HDF5 file for preprocessed training data.')
    train_parser.add_argument('--sub_sample_size', type=int, default=33, help='Size of the sub-samples (default: 33).')
    train_parser.add_argument('--stride', type=int, default=14, help='Stride for sampling (default: 14).')
    train_parser.add_argument('--upscaling', type=int, default=2, help='Upscaling factor (default: 2).')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Process evaluation images.')
    eval_parser.add_argument('image_dir', type=str, help='Directory containing input images for evaluation.')
    eval_parser.add_argument('output_dir', type=str, help='Output HDF5 file for preprocessed evaluation data.')
    eval_parser.add_argument('--sub_sample_size', type=int, default=33, help='Size of the sub-samples (default: 33).')
    eval_parser.add_argument('--stride', type=int, default=14, help='Stride for sampling (default: 14).')
    eval_parser.add_argument('--upscaling', type=int, default=2, help='Upscaling factor (default: 2).')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.image_dir, args.output_dir, args.sub_sample_size, args.stride, args.upscaling)
    elif args.command == 'eval':
        eval(args.image_dir, args.output_dir, args.sub_sample_size, args.stride, args.upscaling)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
