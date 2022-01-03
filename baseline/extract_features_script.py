import argparse
from extract_features import FeatureExtraction


def parse_arguments(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', default="data/train")
    parser.add_argument('--max_seq',default=80)
    parser.add_argument('--token_voc',default="vocab/voc.txt")
    opt = parser.parse_args()
    return opt


def main(opt):
    data_path = opt.data_path
    max_seq = opt.max_seq
    token_voc = opt.token_voc
    
    extractor = FeatureExtraction(data_path,max_seq, token_voc)
    extractor.extract()
    
if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)    
    

