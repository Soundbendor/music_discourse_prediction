from experiment.regressionexperiment import RegressionExperiment
from experiment.classificationexperiment import ClassificationExperiment
from experiment.experiment import ExperimentTypeNotFoundError
from preprocessing.experimentfactory import ExperimentFactory
from preprocessing.dataset import Dataset

import argparse 



def main():
    args = parseargs()
    config = ExperimentFactory(args.config)
    dataset = Dataset(config.get_preprocessing_args(), args.data)
    experiment = get_experiment(dataset, config, args.output)
    experiment.run_experiment()

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyzing features generated off semantic wordlist analyis of social media commentary")
    parser.add_argument('-l', '--enable_logging', dest='log', action='store_true', help='Enables logging of the feature selection')
    parser.add_argument('-c', dest='config', help='The .ini file which holds the experiment description.')
    parser.add_argument('-o', dest='output', default='report.pdf', help='The file which cross validation statistics are output to.\
         If not specified, CV stats will be output to stdout')
    parser.add_argument('data', help='The path to the feature csv')
    args = parser.parse_args()
    
    return args

def get_experiment(ds: Dataset, config: ExperimentFactory, output):
    exp_type = config.get_experiment_type()
    if exp_type == 'regression':
        return RegressionExperiment(ds, config, output)
    elif exp_type == 'classification':
        return ClassificationExperiment(ds, config, output)
    else:
        raise ExperimentTypeNotFoundError("Experiment type specified in the .ini \
            configuration is not a valid experiment type. Please use 'regression' or \
            'classification' experiment types.")


if __name__ == '__main__':
    main()