import argparse
from fscpy import inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model',
                        type=str,
                        required=True,
                        help='path of .pt file for ResNet-18 model')
    parser.add_argument('--sample_img_name',
                        type=str,
                        required=True,
                        help='name of a test image (e.g. 3KR1212BR_48_test.npy)')

    return parser.parse_args()


def main(args):
    inference.predict(trained_model=args.trained_model,
                      sample_img_name=args.sample_img_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
