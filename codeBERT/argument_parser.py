import argparse


def parse_arguments(): 
     parser = argparse.ArgumentParser()
     
     #General arguments
     parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
     parser.add_argument('--epochs', default=4, type=int, help='number of epochs to train for')
     parser.add_argument('--model_path', default="best", help='path to save the model')
     parser.add_argument('--stop', type=int, default=20, help='early stop for training')
     parser.add_argument('--focalloss', type=int, default=0, help='focal loss , 0 to disable')
     parser.add_argument('--alpha', type=int, default=1, help='focal loss alpha')
     parser.add_argument('--gamma', type=int, default=2, help='focal loss gamma')
     parser.add_argument('--weightloss', type=int, default=1, help='weight cross entropy loss')
     
     parser.add_argument('--batch_size', type=int, default=32, help='batch size')
     parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
     parser.add_argument('--statistic_step', type=int, default=100, help='show statistics per a number of steps')
     
     opt = parser.parse_args()
     return opt
