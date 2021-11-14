import argparse


def main():
    args = parseargs()

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature extraction toolkit for music semantic analysis from various social media platforms.")
    parser.add_argument('-i', '--input_dir' dest='input', type=str,
        help = "Path to the directory storing the JSON files for social media data.")
    parser.add_argument('-w', '--wordlist_dir', dest='wordlists', type=str, 
        help = 'Path to the directory containing the wordlists necesscary for feature generation.')
    parser.add_argument('-p', '--processes', type=int, default='16', dest='processes',
        help = 'Number of processes to spawn')
    return parser.parse_args()