import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Triplet network')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')
parser.add_argument('--data-path', default='triplet_data/file', type=str,
                    help='folder on s3 containing data')
parser.add_argument('--save-name', default='model-test', type=str,
                    help='the name of the saved files')

args = parser.parse_args()


print(f'{args.batch_size} is the batchsize to be used for the file saved as {args.save_name}')
