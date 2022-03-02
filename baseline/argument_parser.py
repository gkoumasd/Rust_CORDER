import argparse


def parse_arguments(): 
     parser = argparse.ArgumentParser()
     
     #General arguments
     parser.add_argument('--model', default="BiLSTM", type=str, help='could be either BiLSTM or CNN')
     parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
     parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train for')
     parser.add_argument('--model_path', default="best/bilstm_classification_asm.bin", help='path to save the model')
     parser.add_argument('--patience', type=int, default=20, help='early stop for training')
     
     parser.add_argument('--batch_size', type=int, default=32, help='batch size')
     parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
     parser.add_argument('--statistic_step', type=int, default=100, help='show statistics per a number of steps')
     
     
     parser.add_argument('--train_data', type=str, default='data/train.pickle', help='Name of train data')
     parser.add_argument('--val_data', type=str, default='data/val.pickle', help='Name of val data')
     parser.add_argument('--test_data', type=str, default='data/test.pickle', help='Name of test data')
     
     opt = parser.parse_args()
     return opt

