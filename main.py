import argparse 
import configparser
from experimentfactory import ExperimentFactory
from dataset import Dataset

def main():
    args = parseargs()
    config = ExperimentFactory(args.config)
    X, y = Dataset(config, args.data).split_x_y()

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyzing features generated off semantic wordlist analyis of social media commentary")
    parser.add_argument('-l', '--enable_logging', dest='log', action='store_true', help='Enables logging of the feature selection')
    parser.add_argument('-c', dest='config', help='The .ini file which holds the experiment description.')
    parser.add_argument('data', help='The path to the feature csv')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    main()