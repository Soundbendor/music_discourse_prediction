import argparse 
import configparser

def main():
    args = parseargs()
    config = parseconfig(args.config)
    print(config._sections)

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyzing features generated off semantic wordlist analyis of social media commentary")
    parser.add_argument('-l', '--enable_logging', dest='log', action='store_true', help='Enables logging of the feature selection')
    parser.add_argument('-c', dest='config', help='The .ini file which holds the experiment description.')
    parser.add_argument('data', help='The path to the feature csv')
    args = parser.parse_args()
    
    return args

def parseconfig(fname: str):
    config = configparser.ConfigParser()
    config.read(fname)
    return config

if __name__ == '__main__':
    main()