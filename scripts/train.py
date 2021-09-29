import argparse
from fscpy.train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='batch size')
    parser.add_argument('--print_every',
                        type=int,
                        default=1,
                        help='Indicates how often to print intermediate results (e.g. loss) during each epoch.'
                             'An argument equal to k means print the results after processing k batches.')
    return parser.parse_args()


def main(args):
    train(epochs=args.epochs, batch_size=args.batch_size, print_every=args.print_every)


if __name__ == '__main__':
    args = parse_args()
    main(args)
